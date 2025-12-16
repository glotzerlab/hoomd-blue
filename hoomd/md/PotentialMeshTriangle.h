// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborList.h"
#include "MeshForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/managed_allocator.h"
#include <memory>

#include <vector>

/*! \file PotentialMeshTriangle.h
    \brief Declares PotentialMeshTriangle
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __POTENTIALMESHTRIANGLEPAIR_H__
#define __POTENTIALMESHTRIANGLEPAIR_H__

namespace hoomd
    {
namespace md
    {
/*! Bond potential with evaluator support

    \ingroup computes
*/
template<class evaluator> class PotentialMeshTriangle : public MeshForceCompute
    {
    public:
    //! Param type from evaluator
    typedef typename evaluator::param_type param_type;

    //! Constructs the compute
    PotentialMeshTriangle(std::shared_ptr<SystemDefinition> sysdef, 
		  std::shared_ptr<NeighborList> nlist,
                  std::shared_ptr<MeshDefinition> meshdef);

    //! Destructor
    virtual ~PotentialMeshTriangle();

    /// Set the parameters
    virtual void setParams(unsigned int type, const param_type& param);
    virtual void setParamsPython(std::string type, pybind11::dict param);

    /// Get the parameters
    pybind11::dict getParams(std::string type);

    virtual void setRcut(unsigned int type, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getRCut(std::string type);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(std::string type, Scalar r_cut);
    //! Set ron for a single type pair

    virtual void setNListRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);
    /// Get the r_cut for a single type pair
    Scalar getNListRCut(pybind11::tuple types);
    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setNListRCutPython(pybind11::tuple types, Scalar r_cut);
    //! Set ron for a single type pair

    /// Validate bond type
    void validateType(unsigned int type, std::string action);

    //! Shifting modes that can be applied to the energy
    enum energyShiftMode
        {
        no_shift = 0,
        shift,
        xplor
        };

    //! Set the mode to use for shifting the energy
    void setShiftMode(energyShiftMode mode)
        {
        m_shift_mode = mode;
        }

    void setShiftModePython(std::string mode)
        {
        if (mode == "none")
            {
            m_shift_mode = no_shift;
            }
        else if (mode == "shift")
            {
            m_shift_mode = shift;
            }
        else if (mode == "xplor")
            {
            m_shift_mode = xplor;
            }
        else
            {
            throw std::runtime_error("Invalid energy shift mode.");
            }
        }

    /// Get the mode used for the energy shifting
    std::string getShiftMode()
        {
        switch (m_shift_mode)
            {
        case no_shift:
            return "none";
        case shift:
            return "shift";
        case xplor:
            return "xplor";
        default:
            throw std::runtime_error("Error setting shift mode.");
            }
        }

    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighborlist to use for the computation
    energyShiftMode m_shift_mode; //!< Store the mode with which to handle the energy shift at r_cut
    GPUArray<Scalar> m_rcutsq;    //!< Cutoff radius squared for the potential list per type pair
    GPUArray<Scalar> m_nlistrcutsq;    //!< Cutoff radius squared for the neighbor list per type pair
    /// Per type pair potential parameters
    GPUArray<param_type> m_params;      //!< Bond parameters per type

    /// Track whether we have attached to the Simulation object
    bool m_attached = true;

    /// r_cut (not squared) given to the neighbor list
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist;

    /// Keep track of number of each type of particle
    std::vector<unsigned int> m_num_particles_by_type;

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Actually compute the forces
    void computeForces(uint64_t timestep) override;

    }; // end class PotentialMeshTriangle

template<class evaluator>
PotentialMeshTriangle<evaluator>::PotentialMeshTriangle(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<NeighborList> nlist,
                                        std::shared_ptr<MeshDefinition> meshdef)
    : MeshForceCompute(sysdef, meshdef), m_nlist(nlist), m_shift_mode(no_shift)
    {
    m_exec_conf->msg->notice(5) << "Constructing PotentialMeshTriangle<" << evaluator::getName() << ">"
                                << std::endl;
    assert(m_pdata);
    assert(m_nlist);

    GPUArray<Scalar> rcutsq(m_pdata->getNTypes(), m_exec_conf);
    m_rcutsq.swap(rcutsq);

    // allocate the parameters
    GPUArray<param_type> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    //m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(m_pdata->getNTypes(), m_exec_conf);

    Index2D typpair_idx(m_pdata->getNTypes());
    GPUArray<Scalar> nlistrcutsq(typpair_idx.getNumElements(), m_exec_conf);
    m_nlistrcutsq.swap(nlistrcutsq);


    m_r_cut_nlist = std::make_shared<GPUArray<Scalar>>(typpair_idx.getNumElements(), m_exec_conf);
    nlist->addRCutMatrix(m_r_cut_nlist);

    // get number of each type of particle, needed for energy and pressure correction
    m_num_particles_by_type.resize(m_pdata->getNTypes());
    std::fill(m_num_particles_by_type.begin(), m_num_particles_by_type.end(), 0);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int typeid_i = __scalar_as_int(h_postype.data[i].w);
        m_num_particles_by_type[typeid_i] += 1;
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce number of each type of particle on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      m_num_particles_by_type.data(),
                      m_pdata->getNTypes(),
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

template<class evaluator> PotentialMeshTriangle<evaluator>::~PotentialMeshTriangle()
    {
    m_exec_conf->msg->notice(5) << "Destroying PotentialMeshTriangle<" << evaluator::getName() << ">"
                                << std::endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

/*! \param type Type of the bond to set parameters for
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
void PotentialMeshTriangle<evaluator>::validateType(unsigned int type, std::string action)
    {
    // make sure the type is valid
    if (type >= m_pdata->getNTypes())
        {
        throw std::runtime_error("error in" + action + " for triangle pair potential. invalid type");
        }
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setParams(unsigned int type, const param_type& param)
    {
    // make sure the type is valid
    validateType(type, "setting params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = param;
    }

/*! \param types Type of the bond to set parameters for using string
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
void PotentialMeshTriangle<evaluator>::setParamsPython(std::string type, pybind11::dict param)
    {
    auto itype = m_pdata->getTypeByName(type);
    auto struct_param = param_type(param);
    setParams(itype, struct_param);
    }

/*! \param types Type of the bond to set parameters for using string
    \param param Parameter to set

    Sets the parameters for the potential of a particular bond type
*/
template<class evaluator>
pybind11::dict PotentialMeshTriangle<evaluator>::getParams(std::string type)
    {
    auto itype = m_pdata->getTypeByName(type);
    validateType(itype, "getting params");
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[itype].asDict();
    }


template<class evaluator>
void PotentialMeshTriangle<evaluator>::setRcut(unsigned int type, Scalar rcut)
    {
    validateType(type, "setting r_cut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::readwrite);
        h_rcutsq.data[type] = rcut * rcut;
        }
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setRCutPython(std::string type, Scalar r_cut)
    {
    auto typ = m_pdata->getTypeByName(type);
    setRcut(typ, r_cut);
    }

template<class evaluator> Scalar PotentialMeshTriangle<evaluator>::getRCut(std::string type)
    {
    auto typ = m_pdata->getTypeByName(type);
    validateType(typ, "getting r_cut.");
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);
    return sqrt(h_rcutsq.data[typ]);
    }


template<class evaluator>
void PotentialMeshTriangle<evaluator>::setNListRcut(unsigned int typ1, unsigned int typ2, Scalar rcut)
    {
    validateType(typ1, "setting r_nlistcut");
    validateType(typ2, "setting r_nlistcut");
        {
        // store r_cut**2 for use internally
        ArrayHandle<Scalar> h_nlistrcutsq(m_nlistrcutsq, access_location::host, access_mode::readwrite);
    	Index2D typpair_idx(m_pdata->getNTypes());
	h_nlistrcutsq.data[typpair_idx(typ1, typ2)] = rcut * rcut;
        h_nlistrcutsq.data[typpair_idx(typ2, typ1)] = rcut * rcut;

        // store r_cut unmodified for so the neighbor list knows what particles to include
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::readwrite);
        h_r_cut_nlist.data[typpair_idx(typ1, typ2)] = rcut;
        h_r_cut_nlist.data[typpair_idx(typ2, typ1)] = rcut;
        }
    // notify the neighbor list that we have changed r_cut values
    m_nlist->notifyRCutMatrixChange();
    }

template<class evaluator>
void PotentialMeshTriangle<evaluator>::setNListRCutPython(pybind11::tuple types, Scalar r_cut)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    setNListRcut(typ1, typ2, r_cut);
    }

template<class evaluator> Scalar PotentialMeshTriangle<evaluator>::getNListRCut(pybind11::tuple types)
    {
    auto typ1 = m_pdata->getTypeByName(types[0].cast<std::string>());
    auto typ2 = m_pdata->getTypeByName(types[1].cast<std::string>());
    validateType(typ1, "getting r_cut.");
    validateType(typ2, "getting r_cut.");
    ArrayHandle<Scalar> h_nlistrcutsq(m_nlistrcutsq, access_location::host, access_mode::read);
    Index2D typpair_idx(m_pdata->getNTypes());
    return sqrt(h_nlistrcutsq.data[typpair_idx(typ1, typ2)]);
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
template<class evaluator>
void PotentialMeshTriangle<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // access the neighbor list, particle data, and system box
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
    				    access_location::host,
    				    access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
    				  access_location::host,
    				  access_mode::read);
    //     Index2D nli = m_nlist->getNListIndexer();
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
    				access_location::host,
    				access_mode::read);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);
    size_t virial_pitch = m_virial.getPitch();

    // access the parameters
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rcutsq(m_rcutsq, access_location::host, access_mode::read);

    // Zero data for force calculation
    m_force.zeroFill();
    m_virial.zeroFill();

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<typename Angle::members_t> h_triangles(
        m_mesh_data->getMeshTriangleData()->getMembersArray(),
        access_location::host,
        access_mode::read);


    const unsigned int size = (unsigned int)m_mesh_data->getMeshTriangleData()->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        const typename Angle::members_t& triangle = h_triangles.data[i];

        unsigned int ttag_a = triangle.tag[0];
        assert(ttag_a < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_b = triangle.tag[1];
        assert(ttag_b < m_pdata->getMaximumTag() + 1);
        unsigned int ttag_c = triangle.tag[2];
        assert(ttag_c < m_pdata->getMaximumTag() + 1);

	std::vector<unsigned int> idces(3);

        idces[0] = h_rtag.data[ttag_a];
        idces[1] = h_rtag.data[ttag_b];
        idces[2] = h_rtag.data[ttag_c];

        assert(idces[0] < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idces[1] < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idces[2] < m_pdata->getN() + m_pdata->getNGhosts());

	std::vector<unsigned int> combined_nlist;

        for (unsigned int v_idx = 0; v_idx < 3; v_idx++)
	{
		unsigned int vv_idx = idces[v_idx];
		for (unsigned int idx_i = 0; idx_i < h_n_neigh.data[vv_idx]; idx_i++)
		{
			unsigned int j = h_nlist.data[h_head_list.data[vv_idx] + idx_i];
			bool not_yet_in = true;
			for(unsigned int cn_idx = 0; cn_idx < combined_nlist.size() && not_yet_in; cn_idx++)
				if( combined_nlist[cn_idx] == j)
					not_yet_in = false;
			if (not_yet_in)
				 combined_nlist.push_back(j);
		}
	}

	if( combined_nlist.size() == 0) 
		continue;

        vec3<Scalar> pos_a(h_pos.data[idces[0]].x, h_pos.data[idces[0]].y, h_pos.data[idces[0]].z);
        vec3<Scalar> pos_b(h_pos.data[idces[1]].x, h_pos.data[idces[1]].y, h_pos.data[idces[1]].z);
        vec3<Scalar> pos_c(h_pos.data[idces[2]].x, h_pos.data[idces[2]].y, h_pos.data[idces[2]].z);

        Scalar3 dab;
        dab.x = pos_a.x - pos_b.x;
        dab.y = pos_a.y - pos_b.y;
        dab.z = pos_a.z - pos_b.z;

        Scalar3 dac;
        dac.x = pos_a.x - pos_c.x;
        dac.y = pos_a.y - pos_c.y;
        dac.z = pos_a.z - pos_c.z;

        dab = box.minImage(dab);
        dac = box.minImage(dac);

        Scalar3 dbc = dac-dab;

	Scalar normal_ab = fast::rsqrt(dot(dab,dab));
	Scalar normal_ac = fast::rsqrt(dot(dac,dac));
	Scalar normal_bc = fast::rsqrt(dot(dbc,dbc));

        Scalar3 nab = dab*normal_ab;
        Scalar3 nac = dac*normal_ac;
        Scalar3 nbc = dbc*normal_bc;

        Scalar3 normal_dir;
        normal_dir.x = dab.y * dac.z - dab.z * dac.y;
        normal_dir.y = dab.z * dac.x - dab.x * dac.z;
        normal_dir.z = dab.x * dac.y - dab.y * dac.x;

	Scalar normal_norm = fast::rsqrt(dot(normal_dir,normal_dir));

	normal_dir.x = normal_dir.x*normal_norm;
	normal_dir.y = normal_dir.y*normal_norm;
	normal_dir.z = normal_dir.z*normal_norm;



        // initialize current particle force, potential energy, and virial to 0
        Scalar3 fa = make_scalar3(0, 0, 0);
        Scalar pea = 0.0;

        Scalar3 fb = make_scalar3(0, 0, 0);
        Scalar peb = 0.0;

        Scalar3 fc = make_scalar3(0, 0, 0);
        Scalar pec = 0.0;

        Scalar viriala[6];
        Scalar virialb[6];
        Scalar virialc[6];
        for (unsigned int k = 0; k < 6; k++)
	{
            viriala[k] = Scalar(0.0);
            virialb[k] = Scalar(0.0);
            virialc[k] = Scalar(0.0);
	}


         for (unsigned int k = 0; k < combined_nlist.size(); k++)
                {
                // access the index of this neighbor (MEM TRANSFER: 1 scalar)
                unsigned int j = combined_nlist[k];

		unsigned int typej = __scalar_as_int(h_pos.data[j].w);
		assert(typej < m_pdata->getNTypes());


                const param_type& param = h_params.data[typej];
                Scalar rcutsq = h_rcutsq.data[typej];

		if( rcutsq == 0) continue;


                // calculate dr_ji (MEM TRANSFER: 3 scalars / FLOPS: 3)
        	vec3<Scalar> pj(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        	Scalar3 daj;
        	daj.x = pos_a.x - pj.x;
        	daj.y = pos_a.y - pj.y;
        	daj.z = pos_a.z - pj.z;
                daj = box.minImage(daj);

		Scalar daj_norm = dot(daj,normal_dir);

		Scalar rsq = daj_norm*daj_norm;

		if( rcutsq < rsq) continue;

		Scalar3 dx = normal_dir*daj_norm;

		Scalar Area_a, Area_b, Area_c;

        	Scalar3 dbj;
        	dbj.x = pos_b.x - pj.x;
        	dbj.y = pos_b.y - pj.y;
        	dbj.z = pos_b.z - pj.z;
                dbj = box.minImage(dbj);

		Scalar3 dajbj; 
		dajbj.x = daj.y*dbj.z - daj.z*dbj.y;
		dajbj.y = daj.z*dbj.x - daj.x*dbj.z;
		dajbj.z = daj.x*dbj.y - daj.y*dbj.x;

		Area_c = dot(dajbj,normal_dir);

		Scalar3 dcj;
		dcj.x = pos_c.x - pj.x;
		dcj.y = pos_c.y - pj.y;
		dcj.z = pos_c.z - pj.z;
		dcj = box.minImage(dcj);


		Scalar3 dcjaj; 
		dcjaj.x = dcj.y*daj.z - dcj.z*daj.y;
		dcjaj.y = dcj.z*daj.x - dcj.x*daj.z;
		dcjaj.z = dcj.x*daj.y - dcj.y*daj.x;
		Area_b = dot(dcjaj,normal_dir);

		Scalar3 dbjcj; 
		dbjcj.x = dbj.y*dcj.z - dbj.z*dcj.y;
		dbjcj.y = dbj.z*dcj.x - dbj.x*dcj.z;
		dbjcj.z = dbj.x*dcj.y - dbj.y*dcj.x;

		Area_a = dot(dbjcj,normal_dir);


		if(Area_c <0)
		{
			Area_c = 0;
			if(Area_b <0)
			{
				dx = daj;
				Area_a = 1;
				Area_b = 0;
			}
			else if(Area_a < 0)
			{
				dx = dbj;
				Area_a = 0;
				Area_b = 1;
			}
			else
			{
				Scalar length_ab = dot(daj,nab);
				Scalar ratio_ab = length_ab*normal_ab;

				if( ratio_ab < 0)
				{
					dx = daj;
                                	Area_a = 1;
                                	Area_b = 0;
				}
				else
				{
					if( ratio_ab > 1)
					{
						dx = dbj;
						Area_a = 0;
						Area_b = 1;
					}
					else
					{
						dx = daj - length_ab*nab;
						Area_a = ratio_ab;
						Area_b = 1-ratio_ab;
					}
				}
			}
		}
		else{
			Area_c *= normal_norm;
			if(Area_b <0)
			{
				Area_b = 0;
				if(Area_a < 0)
				{
					dx = dcj;
					Area_a = 0;
					Area_c = 1;
				}
				else
				{
					Scalar length_ac = dot(daj,nac);
					Scalar ratio_ac = length_ac*normal_ac;

					if( ratio_ac < 0)
					{
						dx = daj;
						Area_a = 1;
						Area_c = 0;
					}
					else
					{
						if( ratio_ac > 1)
						{
							dx = dcj;
							Area_a = 0;
							Area_c = 1;
						}
						else
						{
							dx = daj - length_ac*nac;
							Area_a = ratio_ac;
							Area_c = 1-ratio_ac;
						}
					}
				}
			}
			else
			{
				Area_b *= normal_norm;

				if(Area_a <0)
				{
					Scalar length_bc = dot(dbj,nbc);
					Scalar ratio_bc = length_bc*normal_bc;

					Area_a= 0;

					if( ratio_bc < 0)
					{
						dx = dbj;
						Area_b = 1;
						Area_c = 0;
					}
					else
					{
						if( ratio_bc > 1)
						{
							dx = dcj;
							Area_b = 0;
							Area_c = 1;
						}
						else
						{
							dx = dbj - length_bc*nbc;
							Area_b = ratio_bc;
							Area_c = 1-ratio_bc;
						}
					}
				}
				else
				{
					Area_a *= normal_norm;
				}
			}
		}

		rsq = dot(dx,dx);

                bool energy_shift = false;
                if (m_shift_mode == shift)
                    energy_shift = true;

                // compute the force and potential energy
                Scalar force_divr = Scalar(0.0);
                Scalar pair_eng = Scalar(0.0);
                evaluator eval(rsq, rcutsq, param);
                bool evaluated = eval.evalForceAndEnergy(force_divr, pair_eng, energy_shift);
                if (evaluated)
                    {
                    fa += dx * force_divr* Area_a;
                    fb += dx * force_divr* Area_b;
                    fc += dx * force_divr* Area_c;
                    pea += pair_eng * Scalar(0.5)* Area_a;
                    peb += pair_eng * Scalar(0.5)* Area_b;
                    pec += pair_eng * Scalar(0.5)* Area_c;
                    if (compute_virial)
                        {
                        viriala[0] += force_divr * Area_a * pos_a.x * dx.x;
                        viriala[1] += force_divr * Area_a * pos_a.x * dx.y;
                        viriala[2] += force_divr * Area_a * pos_a.x * dx.z;
                        viriala[3] += force_divr * Area_a * pos_a.y * dx.y;
                        viriala[4] += force_divr * Area_a * pos_a.y * dx.z;
                        viriala[5] += force_divr * Area_a * pos_a.z * dx.z;

                        virialb[0] += force_divr * Area_b * pos_b.x * dx.x;
                        virialb[1] += force_divr * Area_b * pos_b.x * dx.y;
                        virialb[2] += force_divr * Area_b * pos_b.x * dx.z;
                        virialb[3] += force_divr * Area_b * pos_b.y * dx.y;
                        virialb[4] += force_divr * Area_b * pos_b.y * dx.z;
                        virialb[5] += force_divr * Area_b * pos_b.z * dx.z;

                        virialc[0] += force_divr * Area_c * pos_c.x * dx.x;
                        virialc[1] += force_divr * Area_c * pos_c.x * dx.y;
                        virialc[2] += force_divr * Area_c * pos_c.x * dx.z;
                        virialc[3] += force_divr * Area_c * pos_c.y * dx.y;
                        virialc[4] += force_divr * Area_c * pos_c.y * dx.z;
                        virialc[5] += force_divr * Area_c * pos_c.z * dx.z;
                        }

	            h_force.data[j].x -= dx.x * force_divr;
	            h_force.data[j].y -= dx.y * force_divr;
	            h_force.data[j].z -= dx.z * force_divr;
	            h_force.data[j].w += pair_eng * Scalar(0.5);
	            if (compute_virial)
	                {
	                h_virial.data[0 * virial_pitch + j]
	            	-= force_divr * pj.x * dx.x;
	                h_virial.data[1 * virial_pitch + j]
	            	-= force_divr * pj.x * dx.y;
	                h_virial.data[2 * virial_pitch + j]
	            	-= force_divr * pj.x * dx.z;
	                h_virial.data[3 * virial_pitch + j]
	            	-= force_divr * pj.y * dx.y;
	                h_virial.data[4 * virial_pitch + j]
	            	-= force_divr * pj.y * dx.z;
	                h_virial.data[5 * virial_pitch + j]
	            	-= force_divr * pj.z * dx.z;
	                }
                    }
		}

            if (idces[0] < m_pdata->getN())
		    {
		    h_force.data[idces[0]].x += fa.x;
		    h_force.data[idces[0]].y += fa.y;
		    h_force.data[idces[0]].z += fa.z;
		    h_force.data[idces[0]].w += pea;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[0]] += viriala[jj];
		    }
            if (idces[1] < m_pdata->getN())
		    {
		    h_force.data[idces[1]].x += fb.x;
		    h_force.data[idces[1]].y += fb.y;
		    h_force.data[idces[1]].z += fb.z;
		    h_force.data[idces[1]].w += peb;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[1]] += virialb[jj];
		    }
            if (idces[2] < m_pdata->getN())
		    {
		    h_force.data[idces[2]].x += fc.x;
		    h_force.data[idces[2]].y += fc.y;
		    h_force.data[idces[2]].z += fc.z;
		    h_force.data[idces[2]].w += pec;
            	    for (int jj = 0; jj < 6; jj++)
                	h_virial.data[jj * virial_pitch + idces[2]] += virialc[jj];
		    }
        }
}

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
template<class evaluator>
CommFlags PotentialMeshTriangle<evaluator>::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    flags[comm_flag::tag] = 1;

    if (evaluator::needsCharge())
        flags[comm_flag::charge] = 1;

    flags |= MeshForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

namespace detail
    {
//! Exports the PotentialMeshTriangle class to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialMeshTriangle(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialMeshTriangle<T>,
                     MeshForceCompute,
                     std::shared_ptr<PotentialMeshTriangle<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<NeighborList>,std::shared_ptr<MeshDefinition>>())
        .def("setParams", &PotentialMeshTriangle<T>::setParamsPython)
        .def("getParams", &PotentialMeshTriangle<T>::getParams)
        .def("setRCut", &PotentialMeshTriangle<T>::setRCutPython)
        .def("getRCut", &PotentialMeshTriangle<T>::getRCut)
        .def("setNlistRCut", &PotentialMeshTriangle<T>::setNListRCutPython)
        .def("getNlistRCut", &PotentialMeshTriangle<T>::getNListRCut)
        .def_property("mode",
                      &PotentialMeshTriangle<T>::getShiftMode,
                      &PotentialMeshTriangle<T>::setShiftModePython);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
