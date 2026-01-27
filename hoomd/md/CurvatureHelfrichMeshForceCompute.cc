// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CurvatureHelfrichMeshForceCompute.h"
//#include "BendingRigidityMeshForceCompute.h"

#include <iostream>
#include <stdexcept>

using namespace std;

// SMALL a relatively small number
#define SMALL Scalar(0.001)


/*! \file CurvatureHelfrichMeshForceCompute.cc
    \brief Contains code for the CurvatureHelfrichMeshForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
CurvatureHelfrichMeshForceCompute::CurvatureHelfrichMeshForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                                   std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing HelfrichMeshForceCompute" << endl;

    // allocate the parameters
    unsigned int n_types = m_mesh_data->getMeshTriangleData()->getNTypes();

    GPUArray<curvature_helfrich_param_t> params(n_types, m_exec_conf);
    m_params.swap(params);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_sigma_dash(m_pdata->getMaxN(), m_exec_conf);

    m_sigma_dash.swap(tmp_sigma_dash);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar3> tmp_norm(m_pdata->getMaxN(), m_exec_conf);

    m_norm.swap(tmp_norm);

    // allocate memory for the per-type normal verctors
    GPUArray<Scalar> tmp_sigma(m_pdata->getMaxN(), m_exec_conf);

    m_sigma.swap(tmp_sigma);

    GPUArray<Scalar> tmp_ival(m_pdata->getMaxN(), m_exec_conf);

    m_ival.swap(tmp_ival);

    GPUArray<Scalar> tmp_iarea(m_pdata->getMaxN(), m_exec_conf);

    m_iarea.swap(tmp_iarea);
    }

CurvatureHelfrichMeshForceCompute::~CurvatureHelfrichMeshForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CurvatureHelfrichMeshForceCompute" << endl;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation

    Sets parameters for the potential of a particular angle type
*/
void CurvatureHelfrichMeshForceCompute::setParams(unsigned int type,
		const curvature_helfrich_param_t& params)
    {
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = params;

    // check for some silly errors a user could make
    if (params.k <= 0)
        m_exec_conf->msg->warning() << "helfrich: specified K <= 0" << endl;
    }

void CurvatureHelfrichMeshForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    auto _params = curvature_helfrich_param_t(params);
    setParams(typ, _params);
    }

pybind11::dict CurvatureHelfrichMeshForceCompute::getParams(std::string type)
    {
    auto typ = m_mesh_data->getMeshBondData()->getTypeByName(type);
    if (typ >= m_mesh_data->getMeshBondData()->getNTypes())
        {
        m_exec_conf->msg->error() << "mesh.helfrich: Invalid mesh type specified" << endl;
        throw runtime_error("Error setting parameters in CurvatureHelfrichMeshForceCompute");
        }
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);
    //pybind11::dict params;
    //params["k"] = h_params.data[typ];
    return h_params.data[typ].asDict();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void CurvatureHelfrichMeshForceCompute::computeForces(uint64_t timestep)
    {
    precomputeParameter();

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();
    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_ival(m_ival, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_iarea(m_iarea, access_location::host, access_mode::read);


    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    assert(h_bonds.data);
    assert(h_sigma.data);
    assert(h_ival.data);
    assert(h_iarea.data);
    assert(h_sigma_dash.data);
    assert(h_norm.data);
    assert(h_triangles.data);

    m_force.zeroFill();
    m_virial.zeroFill();

    const BoxDim& box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    Scalar helfrich_virial[6];
    for (unsigned int i = 0; i < 6; i++)
        helfrich_virial[i] = Scalar(0.0);

    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);
        unsigned int btag_d = bond.tag[3];
        assert(btag_d < m_pdata->getMaximumTag() + 1);

        if (btag_c == btag_d)
            continue;

        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];


        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        Scalar3 dad;
        dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
        dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
        dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

        Scalar3 dbc;
        dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
        dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
        dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

        Scalar3 dbd;
        dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
        dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
        dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);
        Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
        Scalar rad = sqrt(rsqad);

        Scalar rsqbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
        Scalar rbc = sqrt(rsqbc);
        Scalar rsqbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
        Scalar rbd = sqrt(rsqbd);

        Scalar3 nab, nac, nad, nbc, nbd;
        nab = dab / rab;
        nac = dac / rac;
        nad = dad / rad;
        nbc = dbc / rbc;
        nbd = dbd / rbd;

        Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
        if (c_accb > 1.0)
            c_accb = 1.0;
        if (c_accb < -1.0)
            c_accb = -1.0;

        Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
        if (c_addb > 1.0)
            c_addb = 1.0;
        if (c_addb < -1.0)
            c_addb = -1.0;

        Scalar c_abbc = -nab.x * nbc.x - nab.y * nbc.y - nab.z * nbc.z;
        if (c_abbc > 1.0)
            c_abbc = 1.0;
        if (c_abbc < -1.0)
            c_abbc = -1.0;

        Scalar c_abbd = -nab.x * nbd.x - nab.y * nbd.y - nab.z * nbd.z;
        if (c_abbd > 1.0)
            c_abbd = 1.0;
        if (c_abbd < -1.0)
            c_abbd = -1.0;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;
        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar c_baad = nab.x * nad.x + nab.y * nad.y + nab.z * nad.z;
        if (c_baad > 1.0)
            c_baad = 1.0;
        if (c_baad < -1.0)
            c_baad = -1.0;

        Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
        if (inv_s_accb < SMALL)
            inv_s_accb = SMALL;
        inv_s_accb = 1.0 / inv_s_accb;

        Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
        if (inv_s_addb < SMALL)
            inv_s_addb = SMALL;
        inv_s_addb = 1.0 / inv_s_addb;

        Scalar inv_s_abbc = sqrt(1.0 - c_abbc * c_abbc);
        if (inv_s_abbc < SMALL)
            inv_s_abbc = SMALL;
        inv_s_abbc = 1.0 / inv_s_abbc;

        Scalar inv_s_abbd = sqrt(1.0 - c_abbd * c_abbd);
        if (inv_s_abbd < SMALL)
            inv_s_abbd = SMALL;
        inv_s_abbd = 1.0 / inv_s_abbd;

        Scalar inv_s_baac = sqrt(1.0 - c_baac * c_baac);
        if (inv_s_baac < SMALL)
            inv_s_baac = SMALL;
        inv_s_baac = 1.0 / inv_s_baac;

        Scalar inv_s_baad = sqrt(1.0 - c_baad * c_baad);
        if (inv_s_baad < SMALL)
            inv_s_baad = SMALL;
        inv_s_baad = 1.0 / inv_s_baad;

        Scalar cot_accb = c_accb * inv_s_accb;
        Scalar cot_addb = c_addb * inv_s_addb;
        Scalar cot_abbc = c_abbc * inv_s_abbc;
        Scalar cot_abbd = c_abbd * inv_s_abbd;

        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;

        unsigned int meshbond_type = m_mesh_data->getMeshBondData()->getTypeByIndex(i);

	// spontaneous curvature values
	Scalar C0_a = 0.0;
	Scalar C0_b = 0.0;
	Scalar C0_c = 0.0;
	Scalar C0_d = 0.0;
	Scalar eps_kT = 0.0;

	if (h_tag.data[idx_a] > h_params.data[meshbond_type].tag_max)
	{
		C0_a = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_b] > h_params.data[meshbond_type].tag_max)
	{
		C0_b = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_c] > h_params.data[meshbond_type].tag_max)
	{
		C0_c = h_params.data[meshbond_type].C0;
	}
	if (h_tag.data[idx_d] > h_params.data[meshbond_type].tag_max)
	{
		C0_d = h_params.data[meshbond_type].C0;
	}

	if ((h_tag.data[idx_a] > h_params.data[meshbond_type].tag_max) && (h_tag.data[idx_b] > h_params.data[meshbond_type].tag_max))
	{
		eps_kT = h_params.data[meshbond_type].eps_kT;
	}


        Scalar3 sigma_dash_a = h_sigma_dash.data[idx_a] ; // precomputed
        Scalar3 sigma_dash_b = h_sigma_dash.data[idx_b] ; // precomputed
        Scalar3 sigma_dash_c = h_sigma_dash.data[idx_c] ; // precomputed
        Scalar3 sigma_dash_d = h_sigma_dash.data[idx_d] ; // precomputed
							  //
        Scalar3 norm_a = h_norm.data[idx_a] ; // precomputed
        Scalar3 norm_b = h_norm.data[idx_b] ; // precomputed
        Scalar3 norm_c = h_norm.data[idx_c] ; // precomputed
        Scalar3 norm_d = h_norm.data[idx_d] ; // precomputed
					      //
        Scalar rsq_a =norm_a.x*norm_a.x +norm_a.y*norm_a.y+norm_a.z*norm_a.z; // precomputed
        Scalar rsq_b =norm_b.x*norm_b.x +norm_b.y*norm_b.y+norm_b.z*norm_b.z; // precomputed
        Scalar rsq_c =norm_c.x*norm_c.x +norm_c.y*norm_c.y+norm_c.z*norm_c.z; // precomputed
        Scalar rsq_d =norm_d.x*norm_d.x +norm_d.y*norm_d.y+norm_d.z*norm_d.z; // precomputed
									      //
        Scalar r_a =sqrt(rsq_a); // precomputed
        Scalar r_b =sqrt(rsq_b); // precomputed
        Scalar r_c =sqrt(rsq_c); // precomputed
        Scalar r_d =sqrt(rsq_d); // precomputed
				 //
        norm_a /= r_a; // precomputed
        norm_b /= r_b; // precomputed
        norm_c /= r_c; // precomputed
        norm_d /= r_d; // precomputed
				 //

        Scalar sigma_a = h_sigma.data[idx_a]; // precomputed
        Scalar sigma_b = h_sigma.data[idx_b]; // precomputed
        Scalar sigma_c = h_sigma.data[idx_c]; // precomputed
        Scalar sigma_d = h_sigma.data[idx_d]; // precomputed
					      //
        Scalar area_a = h_iarea.data[idx_a]; // precomputed
        Scalar area_b = h_iarea.data[idx_b]; // precomputed
        Scalar area_c = h_iarea.data[idx_c]; // precomputed
        Scalar area_d = h_iarea.data[idx_d]; // precomputed
					     //
	//scalar mean curvature computed from dihedral angles
        Scalar H_a = h_ival.data[idx_a]; // precomputed
        Scalar H_b = h_ival.data[idx_b]; // precomputed
        Scalar H_c = h_ival.data[idx_c]; // precomputed
        Scalar H_d = h_ival.data[idx_d]; // precomputed
					      //
    	//std::cout << "idx:a:" << h_tag.data[idx_a] << " dihedral sum Hi:" << h_ival.data[idx_a] << " sum area ival:" << h_iarea.data[idx_a] << "sigma_a:" << sigma_a << std::endl;
    	//std::cout << "idx:a:" << h_tag.data[idx_a] << " dot sigma dash a:" << dot(sigma_dash_a,sigma_dash_a) << std::endl;
    	//std::cout << "idx:b:" <<h_tag.data[idx_b] << " dihedral sum ival:" << h_ival.data[idx_b] << " sum area ival:" << h_iarea.data[idx_b] << "sigma_a:" << sigma_b << std::endl;
    	//std::cout << "idx:a:" << h_tag.data[idx_a] << " dihedral sum ival:" << h_ival.data[idx_a]  << "sigma_a:" << sigma_a << std::endl;
	//
	// we now compute each of the vectors needed for the forces on each node
	// we theun use the vector quantities, spontaneous curvature, and bending rigidity
	// to calculate the force on each node
	// currently the code assumes constant bending rigidity and varying spontaneous curvature
	//
	Scalar3 Hij_two = (dbc - dbd) * 0.25;
	Scalar3 Kij;
	Scalar3 Sij_one;
	Scalar3 Sij_two;

	Scalar Ha_sum =  (H_a + C0_a);
	Scalar Hb_sum =  (H_b + C0_b);
	Scalar Ha_diff = (H_a - C0_a);
	Scalar Hb_diff = (H_b - C0_b);

	// First we recalculate the dihedral angle for which we'll need the unit normal vectors of the faces
	//
	//we then calculate the cos of the interrior angles
	//
	//use a goemetric identity with the "small" number trick to keep from blowing up to calculate cotangents
	//cotangents of interrior angles are used to calculate Sij1 and Sij2 along with unit normals
	//
	// you then the calculate the per edge force on particle i as follows
	//
	// kappa *  [-1 * ((Hi - spont_Hi) + (Hj - spont_Hj)) * Kij + (1/3*(Hi-spont_Hi)*(Hi+spont_Hi)+2/3*(Hj-spont_Hj)*(Hj+spont_Hj))*Hij_two - (Hi-spont_Hi)*Sij_one - (Hj-spont_Hj)*Sij_two]
	//
	//
	Scalar3 local_norm_one =vec_to_scalar3(cross(vec3<Scalar>(dab),vec3<Scalar>(dac)));
	Scalar3 local_norm_two =vec_to_scalar3(cross(vec3<Scalar>(dad),vec3<Scalar>(dab)));
	Scalar rln_acab = sqrt(local_norm_one.x*local_norm_one.x + local_norm_one.y*local_norm_one.y + local_norm_one.z*local_norm_one.z);
	Scalar rln_adab = sqrt(local_norm_two.x*local_norm_two.x + local_norm_two.y*local_norm_two.y + local_norm_two.z*local_norm_two.z);
	Scalar3 unit_norm_one =local_norm_one / rln_acab;
	Scalar3 unit_norm_two =local_norm_two / rln_adab;

	Scalar cos_dih = dot(unit_norm_one,unit_norm_two); // cosine of the dihedral
	Scalar dihedral_angle = acos(cos_dih); // dihedral angle between the normal vectors of the two triangles bordering edge ij
	Kij = - 0.5 * dihedral_angle * nab ; // 0.5 * dihedral_angle * nba
					    //
	Scalar cot_angle_abc=cot_abbc;// this is used for Sij1
	Scalar cot_angle_abd=cot_abbd;// this is used for Sij1
	Scalar cot_angle_bca=cot_accb;// this is used for Sij2
	Scalar cot_angle_adb=cot_addb;// this is used for Sij2
	//std::cout << "cot_angle_abc: " << cot_angle_abc << " cot_angle_abd: " <<cot_angle_abd  << " cot_angle_bca: " << cot_angle_bca << std::endl;
				    //
	Sij_one=  0.5 * (cot_angle_abc * unit_norm_one + cot_angle_abd * unit_norm_two); //
	Sij_two =  -0.5 * (cot_angle_bca * unit_norm_one + cot_angle_adb * unit_norm_two); //
												 //
	//Sij_one.x=  0.5 * (cot_angle_abc * unit_norm_one.x + cot_angle_abd * unit_norm_two.x); //
	//Sij_one.y=  0.5 * (cot_angle_abc * unit_norm_one.y + cot_angle_abd * unit_norm_two.y); //
	//Sij_one.z=  0.5 * (cot_angle_abc * unit_norm_one.z + cot_angle_abd * unit_norm_two.z); //
	//										       //
	//Sij_two.x =  -0.5 * (cot_angle_bca * unit_norm_one.x + cot_angle_adb * unit_norm_two.x); //
	//Sij_two.y =  -0.5 * (cot_angle_bca * unit_norm_one.y + cot_angle_adb * unit_norm_two.y); //
	//Sij_two.z =  -0.5 * (cot_angle_bca * unit_norm_one.z + cot_angle_adb * unit_norm_two.z); //
	//std::cout << "Sij_one.x: " << Sij_one.x << " Sij_one.y: " << Sij_one.y << " Sij_one.z: " << Sij_one.z << std::endl;
	//std::cout << "Sij_two.x: " << Sij_two.x << " Sij_two.y: " << Sij_two.y << " Sij_two.z: " << Sij_two.z << std::endl;
	std::cout << "Hij_two.x: " << Hij_two.x << " Hij_two.y: " << Hij_two.y << " Hij_two.z: " << Hij_two.z << std::endl;
	std::cout << "Kij.x: " << Kij.x << " Kij.y: " << Kij.y << " Kij.z: " << Kij.z << std::endl;
	std::cout << "Ha diff: " << Ha_diff << " H_asum: " << Ha_sum << std::endl;
											   //
											   //
        Scalar3 Fa;
	Fa =   (Ha_diff * Ha_sum / 3  + 2 * Hb_diff * Hb_sum / 3) * Hij_two -   (Ha_diff + Hb_diff) * Kij  -   (Ha_diff * Sij_one +   Hb_diff * Sij_two);
	//Fa.x = (Ha_diff * Ha_sum / 3  + 2 * Hb_diff * Hb_sum / 3) * Hij_two.x - (Ha_diff + Hb_diff) * Kij.x  - (Ha_diff * Sij_one.x + Hb_diff * Sij_two.x);
	//Fa.y = (Ha_diff * Ha_sum / 3  + 2 * Hb_diff * Hb_sum / 3) * Hij_two.y - (Ha_diff + Hb_diff) * Kij.y  - (Ha_diff * Sij_one.y + Hb_diff * Sij_two.y);
	//Fa.z = (Ha_diff * Ha_sum / 3  + 2 * Hb_diff * Hb_sum / 3) * Hij_two.z - (Ha_diff + Hb_diff) * Kij.z  - (Ha_diff * Sij_one.z + Hb_diff * Sij_two.z);

        Fa *= h_params.data[meshbond_type].k;
        if (compute_virial)
            {
            helfrich_virial[0] = Scalar(1. / 2.) * dab.x * Fa.x; // xx
            helfrich_virial[1] = Scalar(1. / 2.) * dab.y * Fa.x; // xy
            helfrich_virial[2] = Scalar(1. / 2.) * dab.z * Fa.x; // xz
            helfrich_virial[3] = Scalar(1. / 2.) * dab.y * Fa.y; // yy
            helfrich_virial[4] = Scalar(1. / 2.) * dab.z * Fa.y; // yz
            helfrich_virial[5] = Scalar(1. / 2.) * dab.z * Fa.z; // zz
            }

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // do not update ghost particles

        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += Fa.x;
            h_force.data[idx_a].y += Fa.y;
            h_force.data[idx_a].z += Fa.z;
            h_force.data[idx_a].w += (2 * h_params.data[meshbond_type].k
                                      * (Ha_diff * Ha_diff )/area_a + eps_kT);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_a] += helfrich_virial[j];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= Fa.x;
            h_force.data[idx_b].y -= Fa.y;
            h_force.data[idx_b].z -= Fa.z;
            //h_force.data[idx_b].w += (h_params.data[meshbond_type].k * 0.5
            //                         * (Hb_diff * Hb_diff)/area_b  + eps_kT);
            //h_force.data[idx_b].w += (h_params.data[meshbond_type].k * 0.5
            //                         * (Hb_diff * Hb_diff)/area_b  + eps_kT);
            h_force.data[idx_b].w += (2 * h_params.data[meshbond_type].k
                                     * (Hb_diff * Hb_diff)/area_b  + eps_kT);
            for (int j = 0; j < 6; j++)
                h_virial.data[j * virial_pitch + idx_b] += helfrich_virial[j];
            }
        }
    }

void CurvatureHelfrichMeshForceCompute::precomputeParameter()
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    ArrayHandle<typename MeshBond::members_t> h_bonds(
        m_mesh_data->getMeshBondData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();

    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_ival(m_ival, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_iarea(m_iarea, access_location::host, access_mode::read);

    m_sigma.zeroFill();
    m_sigma_dash.zeroFill();
    m_norm.zeroFill();
    m_ival.zeroFill();
    m_iarea.zeroFill();

    // loop over mesh triangles
    const unsigned int size_tri = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size_tri; i++)
        {
        const typename Angle::members_t& triangle = h_triangles.data[i];
        assert(triangle.tag[0] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[1] < m_pdata->getMaximumTag() + 1);
        assert(triangle.tag[2] < m_pdata->getMaximumTag() + 1);

        unsigned int idx_a = h_rtag.data[triangle.tag[0]];
        unsigned int idx_b = h_rtag.data[triangle.tag[1]];
        unsigned int idx_c = h_rtag.data[triangle.tag[2]];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);

        Scalar3 nab, nac;
        nab = dab / rab;
        nac = dac / rac;

        Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;

        if (c_baac > 1.0)
            c_baac = 1.0;
        if (c_baac < -1.0)
            c_baac = -1.0;

        Scalar s_baac = sqrt(1.0 - c_baac * c_baac);
        Scalar inv_s_baac = 1.0 / s_baac;

        Scalar3 dc_drab, dc_drac; // dcos_baac / dr_a
        dc_drab = -nac / rab + c_baac / rab * nab;
        dc_drac = -nab / rac + c_baac / rac * nac;

        Scalar3 ds_drab, ds_drac; // dsin_baac / dr_a
        ds_drab = -c_baac * inv_s_baac * dc_drab;
        ds_drac = -c_baac * inv_s_baac * dc_drac;

        Scalar tri_area = rab * rac * s_baac / 6; // triangle area/3

        //Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        //Scalar rab = sqrt(rsqab);
        //Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        //Scalar rac = sqrt(rsqac);

        Scalar3 local_norm;
        local_norm.x = dac.y * dab.z - dac.z * dab.y;
        local_norm.y = dac.z * dab.x - dac.x * dab.z;
        local_norm.z = dac.x * dab.y - dac.y * dab.x;
        //
        if (idx_a < m_pdata->getN())
            {
            h_norm.data[idx_a].x += local_norm.x ;
            h_norm.data[idx_a].y += local_norm.y ;
            h_norm.data[idx_a].z += local_norm.z ;
	    h_iarea.data[idx_a] += tri_area;
            }
        if (idx_b < m_pdata->getN())
            {
            h_norm.data[idx_b].x += local_norm.x ;
            h_norm.data[idx_b].y += local_norm.y ;
            h_norm.data[idx_b].z += local_norm.z ;
	    h_iarea.data[idx_b] += tri_area;
            }
        if (idx_c < m_pdata->getN())
            {
            h_norm.data[idx_c].x += local_norm.x ;
            h_norm.data[idx_c].y += local_norm.y ;
            h_norm.data[idx_c].z += local_norm.z ;
	    h_iarea.data[idx_c] += tri_area;
            }

        }


    const unsigned int size = (unsigned int)m_mesh_data->getMeshBondData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename MeshBond::members_t& bond = h_bonds.data[i];

        unsigned int btag_a = bond.tag[0];
        assert(btag_a < m_pdata->getMaximumTag() + 1);
        unsigned int btag_b = bond.tag[1];
        assert(btag_b < m_pdata->getMaximumTag() + 1);
        unsigned int btag_c = bond.tag[2];
        assert(btag_c < m_pdata->getMaximumTag() + 1);
        unsigned int btag_d = bond.tag[3];
        assert(btag_d < m_pdata->getMaximumTag() + 1);

        if (btag_c == btag_d)
            continue;

        unsigned int idx_a = h_rtag.data[btag_a];
        unsigned int idx_b = h_rtag.data[btag_b];
        unsigned int idx_c = h_rtag.data[btag_c];
        unsigned int idx_d = h_rtag.data[btag_d];

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

        Scalar3 dad;
        dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
        dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
        dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

        Scalar3 dbc;
        dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
        dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
        dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

        Scalar3 dbd;
        dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
        dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
        dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);
        dad = box.minImage(dad);
        dbc = box.minImage(dbc);
        dbd = box.minImage(dbd);

	// calc dihedral
	//

        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        rac = sqrt(rac);
        Scalar rad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
        rad = sqrt(rad);

        Scalar rbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
        rbc = sqrt(rbc);
        Scalar rbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
        rbd = sqrt(rbd);

        Scalar3 nab, nac, nad, nbc, nbd;
        nab = dab / rab;
        nac = dac / rac;
        nad = dad / rad;
        nbc = dbc / rbc;
        nbd = dbd / rbd;

        Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
        if (c_accb > 1.0)
            c_accb = 1.0;
        if (c_accb < -1.0)
            c_accb = -1.0;

        Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
        if (c_addb > 1.0)
            c_addb = 1.0;
        if (c_addb < -1.0)
            c_addb = -1.0;

        Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
        if (inv_s_accb < SMALL)
            inv_s_accb = SMALL;
        inv_s_accb = 1.0 / inv_s_accb;

        Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
        if (inv_s_addb < SMALL)
            inv_s_addb = SMALL;
        inv_s_addb = 1.0 / inv_s_addb;

        Scalar cot_accb = c_accb * inv_s_accb;
        Scalar cot_addb = c_addb * inv_s_addb;

        Scalar sigma_hat_ab = (cot_accb + cot_addb) / 2;
	//std::cout << "tag: a, b, c, d:" << h_tag.data[idx_a] << "," << h_tag.data[idx_b] << "," << h_tag.data[idx_c] << "," << h_tag.data[idx_d] << "," << std::endl;
	//std::cout << "sigma_hat_ab:" << sigma_hat_ab << std::endl;
	//
	//Scalar3 local_norm_one =vec_to_scalar3(cross(vec3<Scalar>(dab),vec3<Scalar>(dac)));
	//Scalar3 local_norm_two =vec_to_scalar3(cross(vec3<Scalar>(dab),vec3<Scalar>(dad)));
	//Scalar3 local_norm_one =vec_to_scalar3(cross(vec3<Scalar>(dac),vec3<Scalar>(dab)));
	Scalar3 local_norm_one =vec_to_scalar3(cross(vec3<Scalar>(dab),vec3<Scalar>(dac)));
	Scalar3 local_norm_two =vec_to_scalar3(cross(vec3<Scalar>(dad),vec3<Scalar>(dab)));
	Scalar rln_acab = sqrt(local_norm_one.x*local_norm_one.x + local_norm_one.y*local_norm_one.y + local_norm_one.z*local_norm_one.z);
	Scalar rln_adab = sqrt(local_norm_two.x*local_norm_two.x + local_norm_two.y*local_norm_two.y + local_norm_two.z*local_norm_two.z);

        Scalar sigma_a = sigma_hat_ab * rsqab * 0.25;
	Scalar dot_dih = dot(local_norm_one,local_norm_two)/(rln_acab * rln_adab);
	Scalar dihedral = rab*acos(dot_dih)*0.25;
	//std::cout << "rab: " << rab << std::endl;

        if (idx_a < m_pdata->getN())
            {
            h_sigma.data[idx_a] += sigma_a;
	    h_ival.data[idx_a] += dihedral;
            h_sigma_dash.data[idx_a].x += sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_a].y += sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_a].z += sigma_hat_ab * dab.z;
            }

        if (idx_b < m_pdata->getN())
            {
            h_sigma.data[idx_b] += sigma_a;
	    h_ival.data[idx_b] += dihedral;
            h_sigma_dash.data[idx_b].x -= sigma_hat_ab * dab.x;
            h_sigma_dash.data[idx_b].y -= sigma_hat_ab * dab.y;
            h_sigma_dash.data[idx_b].z -= sigma_hat_ab * dab.z;
            }
        }
    }

Scalar CurvatureHelfrichMeshForceCompute::energyDiff(unsigned int idx_a,
                                            unsigned int idx_b,
                                            unsigned int idx_c,
                                            unsigned int idx_d,
                                            unsigned int type_id)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_norm(m_norm, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_ival(m_ival, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_iarea(m_iarea, access_location::host, access_mode::read);

    ArrayHandle<curvature_helfrich_param_t> h_params(m_params, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar C0_a = 0.0;
    Scalar C0_b = 0.0;
    Scalar C0_c = 0.0;
    Scalar C0_d = 0.0;

    Scalar eps_kT_old = 0.0;
    Scalar eps_kT_new = 0.0;

    if (h_rtag.data[idx_a] > h_params.data[type_id].tag_max)
    {
    	C0_a = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_b] > h_params.data[type_id].tag_max)
    {
    	C0_b = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_c] > h_params.data[type_id].tag_max)
    {
    	C0_c = h_params.data[type_id].C0;
    }

    if (h_rtag.data[idx_d] > h_params.data[type_id].tag_max)
    {
    	C0_d = h_params.data[type_id].C0;
    }


    if ((h_rtag.data[idx_a] > h_params.data[type_id].tag_max) && (h_rtag.data[idx_b] > h_params.data[type_id].tag_max))
    {
    	eps_kT_old = h_params.data[type_id].eps_kT;
    }

    if ((h_rtag.data[idx_c] > h_params.data[type_id].tag_max) && (h_rtag.data[idx_d] > h_params.data[type_id].tag_max))
    {
    	eps_kT_new = h_params.data[type_id].eps_kT;
    }


    // calculate d\vec{r}
    Scalar3 dab;
    dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
    dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
    dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;

    Scalar3 dac;
    dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x;
    dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
    dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;

    Scalar3 dad;
    dad.x = h_pos.data[idx_a].x - h_pos.data[idx_d].x;
    dad.y = h_pos.data[idx_a].y - h_pos.data[idx_d].y;
    dad.z = h_pos.data[idx_a].z - h_pos.data[idx_d].z;

    Scalar3 dbc;
    dbc.x = h_pos.data[idx_b].x - h_pos.data[idx_c].x;
    dbc.y = h_pos.data[idx_b].y - h_pos.data[idx_c].y;
    dbc.z = h_pos.data[idx_b].z - h_pos.data[idx_c].z;

    Scalar3 dbd;
    dbd.x = h_pos.data[idx_b].x - h_pos.data[idx_d].x;
    dbd.y = h_pos.data[idx_b].y - h_pos.data[idx_d].y;
    dbd.z = h_pos.data[idx_b].z - h_pos.data[idx_d].z;

    Scalar3 dcd;
    dcd.x = h_pos.data[idx_c].x - h_pos.data[idx_d].x;
    dcd.y = h_pos.data[idx_c].y - h_pos.data[idx_d].y;
    dcd.z = h_pos.data[idx_c].z - h_pos.data[idx_d].z;

    // apply minimum image conventions to all 3 vectors
    dab = box.minImage(dab);
    dac = box.minImage(dac);
    dad = box.minImage(dad);
    dbc = box.minImage(dbc);
    dbd = box.minImage(dbd);
    dcd = box.minImage(dcd);

    Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
    Scalar rab = sqrt(rsqab);
    Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
    Scalar rac = sqrt(rsqac);
    Scalar rsqad = dad.x * dad.x + dad.y * dad.y + dad.z * dad.z;
    Scalar rad = sqrt(rsqad);

    Scalar rsqbc = dbc.x * dbc.x + dbc.y * dbc.y + dbc.z * dbc.z;
    Scalar rbc = sqrt(rsqbc);
    Scalar rsqbd = dbd.x * dbd.x + dbd.y * dbd.y + dbd.z * dbd.z;
    Scalar rbd = sqrt(rsqbd);
    Scalar rsqcd = dcd.x * dcd.x + dcd.y * dcd.y + dcd.z * dcd.z;
    Scalar rcd = sqrt(rsqcd);

    Scalar3 nab, nac, nad, nbc, nbd, ncd;
    nab = dab / rab;
    nac = dac / rac;
    nad = dad / rad;
    nbc = dbc / rbc;
    nbd = dbd / rbd;
    ncd = dcd / rcd;

    Scalar c_accb = nac.x * nbc.x + nac.y * nbc.y + nac.z * nbc.z;
    if (c_accb > 1.0)
        c_accb = 1.0;
    if (c_accb < -1.0)
        c_accb = -1.0;

    Scalar inv_s_accb = sqrt(1.0 - c_accb * c_accb);
    if (inv_s_accb < SMALL)
        inv_s_accb = SMALL;
    inv_s_accb = 1.0 / inv_s_accb;

    Scalar c_addb = nad.x * nbd.x + nad.y * nbd.y + nad.z * nbd.z;
    if (c_addb > 1.0)
        c_addb = 1.0;
    if (c_addb < -1.0)
        c_addb = -1.0;

    Scalar inv_s_addb = sqrt(1.0 - c_addb * c_addb);
    if (inv_s_addb < SMALL)
        inv_s_addb = SMALL;
    inv_s_addb = 1.0 / inv_s_addb;

    Scalar c_abbc = -(nab.x * nbc.x + nab.y * nbc.y + nab.z * nbc.z);
    if (c_abbc > 1.0)
        c_abbc = 1.0;
    if (c_abbc < -1.0)
        c_abbc = -1.0;

    Scalar inv_s_abbc = sqrt(1.0 - c_abbc * c_abbc);
    if (inv_s_abbc < SMALL)
        inv_s_abbc = SMALL;
    inv_s_abbc = 1.0 / inv_s_abbc;

    Scalar c_abbd = -(nab.x * nbd.x + nab.y * nbd.y + nab.z * nbd.z);
    if (c_abbd > 1.0)
        c_abbd = 1.0;
    if (c_abbd < -1.0)
        c_abbd = -1.0;

    Scalar inv_s_abbd = sqrt(1.0 - c_abbd * c_abbd);
    if (inv_s_abbd < SMALL)
        inv_s_abbd = SMALL;
    inv_s_abbd = 1.0 / inv_s_abbd;

    Scalar c_baac = nab.x * nac.x + nab.y * nac.y + nab.z * nac.z;
    if (c_baac > 1.0)
        c_baac = 1.0;
    if (c_baac < -1.0)
        c_baac = -1.0;

    Scalar inv_s_baac = sqrt(1.0 - c_baac * c_baac);
    if (inv_s_baac < SMALL)
        inv_s_baac = SMALL;
    inv_s_baac = 1.0 / inv_s_baac;

    Scalar c_baad = nab.x * nad.x + nab.y * nad.y + nab.z * nad.z;
    if (c_baad > 1.0)
        c_baad = 1.0;
    if (c_baad < -1.0)
        c_baad = -1.0;

    Scalar inv_s_baad = sqrt(1.0 - c_baad * c_baad);
    if (inv_s_baad < SMALL)
        inv_s_baad = SMALL;
    inv_s_baad = 1.0 / inv_s_baad;

    Scalar c_caad = nac.x * nad.x + nac.y * nad.y + nac.z * nad.z;
    if (c_caad > 1.0)
        c_caad = 1.0;
    if (c_caad < -1.0)
        c_caad = -1.0;

    Scalar inv_s_caad = sqrt(1.0 - c_caad * c_caad);
    if (inv_s_caad < SMALL)
        inv_s_caad = SMALL;
    inv_s_caad = 1.0 / inv_s_caad;

    Scalar c_cbbd = nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z;
    if (c_cbbd > 1.0)
        c_cbbd = 1.0;
    if (c_cbbd < -1.0)
        c_cbbd = -1.0;

    Scalar inv_s_cbbd = sqrt(1.0 - c_cbbd * c_cbbd);
    if (inv_s_cbbd < SMALL)
        inv_s_cbbd = SMALL;
    inv_s_cbbd = 1.0 / inv_s_cbbd;

    Scalar c_accd = -(nac.x * ncd.x + nac.y * ncd.y + nac.z * ncd.z);
    if (c_accd > 1.0)
        c_accd = 1.0;
    if (c_accd < -1.0)
        c_accd = -1.0;

    Scalar inv_s_accd = sqrt(1.0 - c_accd * c_accd);
    if (inv_s_accd < SMALL)
        inv_s_accd = SMALL;
    inv_s_accd = 1.0 / inv_s_accd;

    Scalar c_addc = nad.x * ncd.x + nad.y * ncd.y + nad.z * ncd.z;
    if (c_addc > 1.0)
        c_addc = 1.0;
    if (c_addc < -1.0)
        c_addc = -1.0;

    Scalar inv_s_addc = sqrt(1.0 - c_addc * c_addc);
    if (inv_s_addc < SMALL)
        inv_s_addc = SMALL;
    inv_s_addc = 1.0 / inv_s_addc;

    Scalar c_bccd = -(nbc.x * ncd.x + nbc.y * ncd.y + nbc.z * ncd.z);
    if (c_bccd > 1.0)
        c_bccd = 1.0;
    if (c_bccd < -1.0)
        c_bccd = -1.0;

    Scalar inv_s_bccd = sqrt(1.0 - c_bccd * c_bccd);
    if (inv_s_bccd < SMALL)
        inv_s_bccd = SMALL;
    inv_s_bccd = 1.0 / inv_s_bccd;

    Scalar c_bddc = nbd.x * ncd.x + nbd.y * ncd.y + nbd.z * ncd.z;
    if (c_bddc > 1.0)
        c_bddc = 1.0;
    if (c_bddc < -1.0)
        c_bddc = -1.0;

    Scalar inv_s_bddc = sqrt(1.0 - c_bddc * c_bddc);
    if (inv_s_bddc < SMALL)
        inv_s_bddc = SMALL;
    inv_s_bddc = 1.0 / inv_s_bddc;

    Scalar cot_accb = c_accb * inv_s_accb;
    Scalar cot_addb = c_addb * inv_s_addb;
    Scalar cot_baac = c_baac * inv_s_baac;
    Scalar cot_baad = c_baad * inv_s_baad;
    Scalar cot_abbc = c_abbc * inv_s_abbc;
    Scalar cot_abbd = c_abbd * inv_s_abbd;

    Scalar cot_caad = c_caad * inv_s_caad;
    Scalar cot_cbbd = c_cbbd * inv_s_cbbd;
    Scalar cot_accd = c_accd * inv_s_accd;
    Scalar cot_addc = c_addc * inv_s_addc;
    Scalar cot_bccd = c_bccd * inv_s_bccd;
    Scalar cot_bddc = c_bddc * inv_s_bddc;

    Scalar sigma_hat_ab = -(cot_accb + cot_addb) * 0.5;
    Scalar sigma_hat_cd = (cot_caad + cot_cbbd) * 0.5;
    Scalar sigma_hat_ac = (cot_addc - cot_abbc) * 0.5;
    Scalar sigma_hat_ad = (cot_accd - cot_abbd) * 0.5;
    Scalar sigma_hat_bc = (cot_bddc - cot_baac) * 0.5;
    Scalar sigma_hat_bd = (cot_bccd - cot_baad) * 0.5;

    Scalar sigma_a = h_sigma.data[idx_a]; // precomputed
    Scalar sigma_b = h_sigma.data[idx_b]; // precomputed
    Scalar sigma_c = h_sigma.data[idx_c]; // precomputed
    Scalar sigma_d = h_sigma.data[idx_d]; // precomputed

    m_sigma_diff_a = (sigma_hat_ab * rsqab + sigma_hat_ac * rsqac + sigma_hat_ad * rsqad) * 0.25;
    m_sigma_diff_b = (sigma_hat_ab * rsqab + sigma_hat_bc * rsqbc + sigma_hat_bd * rsqbd) * 0.25;
    m_sigma_diff_c = (sigma_hat_ac * rsqac + sigma_hat_bc * rsqbc + sigma_hat_cd * rsqcd) * 0.25;
    m_sigma_diff_d = (sigma_hat_ad * rsqad + sigma_hat_bd * rsqbd + sigma_hat_cd * rsqcd) * 0.25;

    Scalar sigma_a_n = sigma_a + m_sigma_diff_a;
    Scalar sigma_b_n = sigma_b + m_sigma_diff_b;
    Scalar sigma_c_n = sigma_c + m_sigma_diff_c;
    Scalar sigma_d_n = sigma_d + m_sigma_diff_d;

    Scalar3 sigma_dash_a = h_sigma_dash.data[idx_a]; // precomputed
    Scalar3 sigma_dash_b = h_sigma_dash.data[idx_b]; // precomputed
    Scalar3 sigma_dash_c = h_sigma_dash.data[idx_c]; // precomputed
    Scalar3 sigma_dash_d = h_sigma_dash.data[idx_d]; // precomputed

    m_sigma_dash_diff_a = sigma_hat_ab * dab + sigma_hat_ac * dac + sigma_hat_ad * dad;
    m_sigma_dash_diff_b = -sigma_hat_ab * dab + sigma_hat_bc * dbc + sigma_hat_bd * dbd;
    m_sigma_dash_diff_c = -sigma_hat_ac * dac - sigma_hat_bc * dbc + sigma_hat_cd * dcd;
    m_sigma_dash_diff_d = -sigma_hat_ad * dad - sigma_hat_bd * dbd - sigma_hat_cd * dcd;

    Scalar3 sigma_dash_a_n = sigma_dash_a + m_sigma_dash_diff_a;
    Scalar3 sigma_dash_b_n = sigma_dash_b + m_sigma_dash_diff_b;
    Scalar3 sigma_dash_c_n = sigma_dash_c + m_sigma_dash_diff_c;
    Scalar3 sigma_dash_d_n = sigma_dash_d + m_sigma_dash_diff_d;

    Scalar sq_V_a=       dot(sigma_dash_a, sigma_dash_a)       ;
    Scalar sq_V_b=       dot(sigma_dash_b, sigma_dash_b);
    Scalar sq_V_c=       dot(sigma_dash_c, sigma_dash_c);
    Scalar sq_V_d=       dot(sigma_dash_d, sigma_dash_d);

    Scalar V_a=sqrt(sq_V_a);
    Scalar V_b=sqrt(sq_V_b);
    Scalar V_c=sqrt(sq_V_c);
    Scalar V_d=sqrt(sq_V_d);

    Scalar sq_V_a_n= dot(sigma_dash_a_n, sigma_dash_a_n);
    Scalar sq_V_b_n= dot(sigma_dash_b_n, sigma_dash_b_n);
    Scalar sq_V_c_n= dot(sigma_dash_c_n, sigma_dash_c_n);
    Scalar sq_V_d_n= dot(sigma_dash_d_n, sigma_dash_d_n);

    Scalar V_a_n=sqrt(sq_V_a_n);
    Scalar V_b_n=sqrt(sq_V_b_n);
    Scalar V_c_n=sqrt(sq_V_c_n);
    Scalar V_d_n=sqrt(sq_V_d_n);

    // we now compute the sign of this contribution to the curvature based on the normal
    // of the face of the triangle associated with this bond - question of if this norm is enough
    // to calculate normal direction of the vertex or if this has to be accomplished with precompute
    // parameters
    Scalar3 norm_a = h_norm.data[idx_a] ; // precomputed
    Scalar3 norm_b = h_norm.data[idx_b] ; // precomputed
    Scalar3 norm_c = h_norm.data[idx_c] ; // precomputed
    Scalar3 norm_d = h_norm.data[idx_d] ; // precomputed
					  //
    Scalar3 norm_a_n = h_norm.data[idx_a] ; // precomputed
    Scalar3 norm_b_n = h_norm.data[idx_b] ; // precomputed
    Scalar3 norm_c_n = h_norm.data[idx_c] ; // precomputed
    Scalar3 norm_d_n = h_norm.data[idx_d] ; // precomputed
					      //
    Scalar3 norm_t1;
    Scalar3 norm_t2;
    Scalar3 norm_t3;
    Scalar3 norm_t4;

    norm_t1.x = dac.y * dab.z - dac.z * dab.y;
    norm_t1.y = dac.z * dab.x - dac.x * dab.z;
    norm_t1.z = dac.x * dab.y - dac.y * dab.x;

    norm_t2.x = dab.y * dad.z - dab.z * dad.y;
    norm_t2.y = dab.z * dad.x - dab.x * dad.z;
    norm_t2.z = dab.x * dad.y - dab.y * dad.x;

    norm_t3.x = dac.y * dad.z - dac.z * dad.y;
    norm_t3.y = dac.z * dad.x - dac.x * dad.z;
    norm_t3.z = dac.x * dad.y - dac.y * dad.x;

    norm_t4.x = dbd.y * dbc.z - dbd.z * dbc.y;
    norm_t4.y = dbd.z * dbc.x - dbd.x * dbc.z;
    norm_t4.z = dbd.x * dbc.y - dbd.y * dbc.x;

    norm_a_n += (norm_t3 - norm_t1 - norm_t2);
    norm_b_n += (norm_t4 - norm_t1 - norm_t2);
    norm_c_n += (norm_t4 + norm_t3 - norm_t1);
    norm_d_n += (norm_t4 + norm_t3 - norm_t2);

    Scalar dot_norm_a = dot(norm_a,sigma_dash_a);
    Scalar dot_norm_b = dot(norm_b,sigma_dash_b);
    Scalar dot_norm_c = dot(norm_c,sigma_dash_c);
    Scalar dot_norm_d = dot(norm_d,sigma_dash_d);

    Scalar dot_norm_a_n = dot(norm_a_n,sigma_dash_a_n);
    Scalar dot_norm_b_n = dot(norm_b_n,sigma_dash_b_n);
    Scalar dot_norm_c_n = dot(norm_c_n,sigma_dash_c_n);
    Scalar dot_norm_d_n = dot(norm_d_n,sigma_dash_d_n);

    if (dot_norm_a < 0.0)
        {
        V_a *= -1.0;
        }
    if (dot_norm_b < 0.0)
        {
        V_b *= -1.0;
        }
    if (dot_norm_c < 0.0)
        {
        V_c *= -1.0;
        }
    if (dot_norm_d < 0.0)
        {
        V_d *= -1.0;
        }

    if (dot_norm_a_n < 0.0)
        {
        V_a_n *= -1.0;
        }
    if (dot_norm_b_n < 0.0)
        {
        V_b_n *= -1.0;
        }
    if (dot_norm_c_n < 0.0)
        {
        V_c_n *= -1.0;
        }
    if (dot_norm_d_n < 0.0)
        {
        V_d_n *= -1.0;
        }

    Scalar C0_sq_a = 4 * C0_a * C0_a;
    Scalar C0_sq_b = 4 * C0_b * C0_b;
    Scalar C0_sq_c = 4 * C0_c * C0_c;
    Scalar C0_sq_d = 4 * C0_d * C0_d;


    Scalar energy_old =(sq_V_a/ sigma_a - 4 * C0_a * V_a + C0_sq_a* sigma_a);
    energy_old +=(sq_V_b/ sigma_b - 4 * C0_b * V_b + C0_sq_b* sigma_b);
    energy_old +=(sq_V_c/ sigma_c - 4 * C0_c * V_c + C0_sq_c* sigma_c);
    energy_old +=(sq_V_d/ sigma_d - 4 * C0_d * V_d + C0_sq_d* sigma_d);

    Scalar energy_new = (sq_V_a_n/ sigma_a_n - 4 * C0_a * V_a_n + C0_sq_a) ;
    energy_new +=      ((sq_V_b_n/ sigma_b_n - 4 * C0_b * V_b_n + C0_sq_b) );
    energy_new +=      ((sq_V_c_n/ sigma_c_n - 4 * C0_c * V_c_n + C0_sq_c) );
    energy_new +=      ((sq_V_d_n/ sigma_d_n - 4 * C0_d * V_d_n + C0_sq_d) );

    if (energy_new < 0)
        return DBL_MAX;

    return h_params.data[type_id].k * 0.5 * (energy_new - energy_old) + eps_kT_new - eps_kT_old;
    }

void CurvatureHelfrichMeshForceCompute::postcomputeParameter(unsigned int idx_a,
                                                    unsigned int idx_b,
                                                    unsigned int idx_c,
                                                    unsigned int idx_d,
                                                    unsigned int type_id)
    {
    ArrayHandle<Scalar> h_sigma(m_sigma, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_sigma_dash(m_sigma_dash, access_location::host, access_mode::readwrite);

    h_sigma.data[idx_a] += m_sigma_diff_a;
    h_sigma.data[idx_b] += m_sigma_diff_b;
    h_sigma.data[idx_c] += m_sigma_diff_c;
    h_sigma.data[idx_d] += m_sigma_diff_d;

    h_sigma_dash.data[idx_a] += m_sigma_dash_diff_a;
    h_sigma_dash.data[idx_b] += m_sigma_dash_diff_b;
    h_sigma_dash.data[idx_c] += m_sigma_dash_diff_c;
    h_sigma_dash.data[idx_d] += m_sigma_dash_diff_d;
    }

namespace detail
    {
void export_CurvatureHelfrichMeshForceCompute(pybind11::module& m)
    {
    pybind11::class_<CurvatureHelfrichMeshForceCompute,
                     MeshForceCompute,
                     std::shared_ptr<CurvatureHelfrichMeshForceCompute>>(m, "CurvatureHelfrichMeshForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>())
        .def("setParams", &CurvatureHelfrichMeshForceCompute::setParamsPython)
        .def("getParams", &CurvatureHelfrichMeshForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
