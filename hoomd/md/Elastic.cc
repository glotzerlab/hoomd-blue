// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Elastic.h"

namespace hoomd
    {
namespace md
    {

Elastic::Elastic(std::shared_ptr<SystemDefinition> sysdef,
            std::shared_ptr<TetrahedronData> meshdef,
            pybind11::array_t<Scalar> reference_positions,
            pybind11::array_t<unsigned int> reference_tags): ForceCompute(sysdef){
            m_tetrahedron_data = meshdef;
            setReference(reference_positions, reference_tags);

            GPUArray<ElasticCoefficients> params(m_tetrahedron_data->getNTypes(), m_exec_conf);
            m_params.swap(params);
            }

void Elastic::setParams(unsigned int type, const ElasticCoefficients& params)
	
	{
	// make sure the type is valid
	if (type >= m_tetrahedron_data->getNTypes())
		{
		throw std::runtime_error("Invalid tetrahedron type.");
		}

	// set parameters in m_params
	ArrayHandle<ElasticCoefficients> h_params(m_params, access_location::host, access_mode::overwrite);
	h_params.data[type] = params;
	}

void Elastic::setParamsPython(std::string type_name, pybind11::dict params)
	{
	auto type_id = m_tetrahedron_data->getTypeByName(type_name);
	
	ElasticCoefficients tetra_params;
	tetra_params.elastic_coeff_1 = params["C_xxxx"].cast<Scalar>();
	tetra_params.elastic_coeff_2 = params["C_xxyy"].cast<Scalar>();
	tetra_params.elastic_coeff_3 = params["C_xyxy"].cast<Scalar>();
	
	setParams(type_id, tetra_params);
	}		

pybind11::dict Elastic::getParams(std::string type_name)
	{
	auto type_id = m_tetrahedron_data->getTypeByName(type_name);
	// make sure the type is valid
	if (type_id >= m_tetrahedron_data->getNTypes())
		{
		throw std::runtime_error("Invalid tetrahedron type.");
		}	
	
	ArrayHandle<ElasticCoefficients> h_params(m_params, access_location::host, access_mode::read);
	auto val = h_params.data[type_id];
	pybind11::dict params;
	params["C_xxxx"] = val.elastic_coeff_1;
	params["C_xxyy"] = val.elastic_coeff_2;
	params["C_xyxy"] = val.elastic_coeff_3;
	return params;
	}

void Elastic::setReference(pybind11::array_t<Scalar> reference_positions,
                           pybind11::array_t<unsigned int> reference_tags)
    {
    // Validate that arrays from pybind are of correct shape
    auto reference_positions_indexer = reference_positions.unchecked<2>();
    auto reference_tags_indexer = reference_tags.unchecked<1>();

    const auto N_reference_pos = reference_positions_indexer.shape(0);

    const auto& box = m_pdata->getGlobalBox();

    if (reference_tags_indexer.shape(0) != reference_positions_indexer.shape(0))
        {
        throw std::invalid_argument("The array of tetrahedron vertex positions and tags must have "
                                    "the same number of rows (N_particles).");
        }
    if (reference_positions_indexer.shape(1) != 3)
        {
        throw std::invalid_argument(
            "The array of tetrahedron vertex positions must have three columns.");
        }
    // Initialize m_reference_vertex_displacements
    GPUArray<vec3<Scalar>> reference_vertex_displacements(3 * m_tetrahedron_data->getN(),
                                                          m_exec_conf);
    m_reference_vertex_displacements.swap(reference_vertex_displacements);
    ArrayHandle<vec3<Scalar>> h_reference_vertex_displacements(m_reference_vertex_displacements,
                                                               access_location::host,
                                                               access_mode::overwrite);

    // Initialize m_reference_inv_matrix
    GPUArray<vec3<Scalar>> reference_inv_matrix(3 * m_tetrahedron_data->getN(), m_exec_conf);
    m_reference_inv_matrix.swap(reference_inv_matrix);
    ArrayHandle<vec3<Scalar>> h_reference_inv_matrix(m_reference_inv_matrix,
                                                     access_location::host,
                                                     access_mode::overwrite);

    const auto n_tetrahedra = static_cast<unsigned int>(m_tetrahedron_data->getN());

    std::unordered_map<unsigned int, unsigned int> index_map;

    for (unsigned int i = 0; i < N_reference_pos; i++)
        {
        index_map[reference_tags_indexer(i)] = i;
        }

    m_matrix_indexer = Index2D(n_tetrahedra, 3);

    for (unsigned int i = 0; i < n_tetrahedra; i++)
        {
        const auto& tetrahedron = m_tetrahedron_data->getMembersByIndex(i);

        auto particle_index_0 = index_map[tetrahedron.tag[0]];
        auto particle_index_1 = index_map[tetrahedron.tag[1]];
        auto particle_index_2 = index_map[tetrahedron.tag[2]];
        auto particle_index_3 = index_map[tetrahedron.tag[3]];
        // TODO: come back late to add error checking

        auto r0 = vec3<Scalar>(reference_positions_indexer(particle_index_0, 0),
                               reference_positions_indexer(particle_index_0, 1),
                               reference_positions_indexer(particle_index_0, 2));

        auto r1 = vec3<Scalar>(reference_positions_indexer(particle_index_1, 0),
                               reference_positions_indexer(particle_index_1, 1),
                               reference_positions_indexer(particle_index_1, 2));

        auto r2 = vec3<Scalar>(reference_positions_indexer(particle_index_2, 0),
                               reference_positions_indexer(particle_index_2, 1),
                               reference_positions_indexer(particle_index_2, 2));

        auto r3 = vec3<Scalar>(reference_positions_indexer(particle_index_3, 0),
                               reference_positions_indexer(particle_index_3, 1),
                               reference_positions_indexer(particle_index_3, 2));

        h_reference_vertex_displacements.data[m_matrix_indexer(i, 0)] = box.minImage(r0 - r3);
        h_reference_vertex_displacements.data[m_matrix_indexer(i, 1)] = box.minImage(r1 - r3);
        h_reference_vertex_displacements.data[m_matrix_indexer(i, 2)] = box.minImage(r2 - r3);

        Eigen::Matrix<Scalar, 3, 3> reference_vertex_displacement_matrix;

        for (unsigned int column = 0; column < 3; column++)
            {
            reference_vertex_displacement_matrix(0, column)
                = h_reference_vertex_displacements.data[m_matrix_indexer(i, column)].x;
            reference_vertex_displacement_matrix(1, column)
                = h_reference_vertex_displacements.data[m_matrix_indexer(i, column)].y;
            reference_vertex_displacement_matrix(2, column)
                = h_reference_vertex_displacements.data[m_matrix_indexer(i, column)].z;
            }

        auto reference_vertex_displacement_matrix_inverse
            = reference_vertex_displacement_matrix.inverse();

        for (unsigned int column = 0; column < 3; column++)
            {
            h_reference_inv_matrix.data[m_matrix_indexer(i, column)].x
                = reference_vertex_displacement_matrix_inverse(0, column);
            h_reference_inv_matrix.data[m_matrix_indexer(i, column)].y
                = reference_vertex_displacement_matrix_inverse(1, column);
            h_reference_inv_matrix.data[m_matrix_indexer(i, column)].z
                = reference_vertex_displacement_matrix_inverse(2, column);
            }
        }
    }

void Elastic::computeForces(uint64_t timestep)
    {
    // Create
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_particle_rtag(m_pdata->getRTags(),
                                              access_location::host,
                                              access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);

    ArrayHandle<vec3<Scalar>> h_reference_vertex_displacements(m_reference_vertex_displacements,
                                                               access_location::host,
                                                               access_mode::read);

    ArrayHandle<vec3<Scalar>> h_reference_inv_matrix(m_reference_inv_matrix,
                                                     access_location::host,
                                                     access_mode::read);

	ArrayHandle<ElasticCoefficients> h_params(m_params, 
											  access_location::host,
											  access_mode::read);

    // Zero data for force calculation
    m_force.zeroFill();
    m_virial.zeroFill();

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<typename TetrahedronData::members_t> h_tetrahedron(
        m_tetrahedron_data->getMembersArray(),
        access_location::host,
        access_mode::read);
    ArrayHandle<typeval_t> h_typeval(m_tetrahedron_data->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    unsigned int max_local = m_pdata->getN() + m_pdata->getNGhosts();

    const size_t n_tetrahedra = m_tetrahedron_data->getN();

    // Calculate forces for each tetrahedron

    for (unsigned int i = 0; i < n_tetrahedra; i++)
        {
        const auto& tetrahedron = h_tetrahedron.data[i];
        unsigned int idx_0 = h_particle_rtag.data[tetrahedron.tag[0]];
        unsigned int idx_1 = h_particle_rtag.data[tetrahedron.tag[1]];
        unsigned int idx_2 = h_particle_rtag.data[tetrahedron.tag[2]];
        unsigned int idx_3 = h_particle_rtag.data[tetrahedron.tag[3]];

        if (idx_0 >= max_local || idx_1 >= max_local || idx_2 >= max_local || idx_3 >= max_local)
            {
            std::ostringstream stream;
            stream << "Error: tetrahedron " << tetrahedron.tag[0] << " " << tetrahedron.tag[1]
                   << " " << tetrahedron.tag[2] << " " << tetrahedron.tag[3] << " is incomplete.";
            throw std::runtime_error(stream.str());
            }

        // Step 1: Get displacements
        vec3<Scalar> pos0
            = vec3<Scalar>(h_pos.data[idx_0].x, h_pos.data[idx_0].y, h_pos.data[idx_0].z);
        vec3<Scalar> pos1
            = vec3<Scalar>(h_pos.data[idx_1].x, h_pos.data[idx_1].y, h_pos.data[idx_1].z);
        vec3<Scalar> pos2
            = vec3<Scalar>(h_pos.data[idx_2].x, h_pos.data[idx_2].y, h_pos.data[idx_2].z);
        vec3<Scalar> pos3
            = vec3<Scalar>(h_pos.data[idx_3].x, h_pos.data[idx_3].y, h_pos.data[idx_3].z);

        vec3<Scalar> vertex_displacement_0 = box.minImage(pos0 - pos3);
        vec3<Scalar> vertex_displacement_1 = box.minImage(pos1 - pos3);
        vec3<Scalar> vertex_displacement_2 = box.minImage(pos2 - pos3);

        vec3<Scalar> reference_vertex_displacement_0
            = h_reference_vertex_displacements.data[m_matrix_indexer(i, 0)];
        vec3<Scalar> reference_vertex_displacement_1
            = h_reference_vertex_displacements.data[m_matrix_indexer(i, 1)];
        vec3<Scalar> reference_vertex_displacement_2
            = h_reference_vertex_displacements.data[m_matrix_indexer(i, 2)];

        vec3<Scalar> u = vec3<Scalar>(vertex_displacement_0.x - reference_vertex_displacement_0.x,
                                      vertex_displacement_1.x - reference_vertex_displacement_1.x,
                                      vertex_displacement_2.x - reference_vertex_displacement_2.x);

        vec3<Scalar> v = vec3<Scalar>(vertex_displacement_0.y - reference_vertex_displacement_0.y,
                                      vertex_displacement_1.y - reference_vertex_displacement_1.y,
                                      vertex_displacement_2.y - reference_vertex_displacement_2.y);

        vec3<Scalar> w = vec3<Scalar>(vertex_displacement_0.z - reference_vertex_displacement_0.z,
                                      vertex_displacement_1.z - reference_vertex_displacement_1.z,
                                      vertex_displacement_2.z - reference_vertex_displacement_2.z);

        // Step 2: Calculate Mapping coefficients (a, b, c)

        vec3<Scalar> inverse_matrix_row_0 = h_reference_inv_matrix.data[m_matrix_indexer(i, 0)];
        vec3<Scalar> inverse_matrix_row_1 = h_reference_inv_matrix.data[m_matrix_indexer(i, 1)];
        vec3<Scalar> inverse_matrix_row_2 = h_reference_inv_matrix.data[m_matrix_indexer(i, 2)];

        vec3<Scalar> a = vec3<Scalar>(dot(inverse_matrix_row_0, u),
                                      dot(inverse_matrix_row_1, u),
                                      dot(inverse_matrix_row_2, u));
        vec3<Scalar> b = vec3<Scalar>(dot(inverse_matrix_row_0, v),
                                      dot(inverse_matrix_row_1, v),
                                      dot(inverse_matrix_row_2, v));
        vec3<Scalar> c = vec3<Scalar>(dot(inverse_matrix_row_0, w),
                                      dot(inverse_matrix_row_1, w),
                                      dot(inverse_matrix_row_2, w));

        // Step 3: Calculate Strains (epsilons)

        Scalar strain_tensor[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

        // double check index changes from Matthew's code
        strain_tensor[0][0] = a.x + 0.5 * (a.x * a.x + b.x * b.x + c.x * c.x);
        strain_tensor[0][0] = a.x + 0.5 * (a.x * a.x + b.x * b.x + c.x * c.x);
        strain_tensor[1][1] = b.y + 0.5 * (a.y * a.y + b.y * b.y + c.y * c.y);
        strain_tensor[2][2] = c.z + 0.5 * (a.z * a.z + b.z * b.z + c.z * c.z);
        strain_tensor[0][1] = 0.5 * (a.y + b.x + a.x * a.y + b.x * b.y + c.x * c.y);
        strain_tensor[1][0] = strain_tensor[0][1];
        strain_tensor[0][2] = 0.5 * (a.z + c.x + a.x * a.z + b.x * b.z + c.x * c.z);
        strain_tensor[2][0] = strain_tensor[0][2];
        strain_tensor[1][2] = 0.5 * (b.z + c.y + a.y * a.z + b.y * b.z + c.y * c.z);
        strain_tensor[2][1] = strain_tensor[1][2];

        // Step 4: Calculate Forces
		// The three components of these vectors are the forces of the first, second and third tetrahedron. 
		// The fourth has 0 displacement
		vec3<Scalar> fx = vec3<Scalar>(0,0,0);
		vec3<Scalar> fy = vec3<Scalar>(0,0,0);
		vec3<Scalar> fz = vec3<Scalar>(0,0,0);


		ElasticCoefficients elastic_param_c = h_params.data[m_tetrahedron_data->getTypeByIndex(i)];

		auto minv1 = inverse_matrix_row_0;
		auto minv2 = inverse_matrix_row_1;
		auto minv3 = inverse_matrix_row_2;


		fx += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[0][0] * minv1 * (1.0 + a.x);
		fx += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[1][1] * minv2 * a.y;
		fx += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[2][2] * minv3 * a.z;
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv2 * a.y;
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv1 * (1.0 + a.x);
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv3 * a.z;
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv2 * a.y;
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv1 * (1.0 + a.x);
		fx += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv3 * a.z;
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv2 * (1.0 + a.x);
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv1 * a.y;
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv3 * a.y;
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv2 * a.z;
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv3 * (1.0 + a.x);
		fx += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv1 * a.z;

		fy += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[0][0] * minv1 * b.x;
		fy += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[1][1] * minv2 * (1.0+b.y);
		fy += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[2][2] * minv3 * b.z;
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv2 * (1.0 + b.y);
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv1 * b.x;
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv3 * b.z;
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv2 * (1.0 + b.y);
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv1 * b.x;
		fy += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv3 * b.z;
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv2 * b.x;
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv1 * (1.0 + b.y);
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv3 * (1.0 + b.y);
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv2 * b.z;
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv3 * b.x;
		fy += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv1 * a.z;
		
		fz += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[0][0] * minv1 * c.x;
		fz += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[1][1] * minv2 * c.y;
		fz += -2.0 * elastic_param_c.elastic_coeff_1 * strain_tensor[2][2] * minv3 * (1.0 + c.z);
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv2 * c.y;
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv1 * c.x;
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[1][1] * minv3 * (1.0 + c.z);
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv2 * c.y;
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[2][2] * minv1 * c.x;
		fz += -2.0 * elastic_param_c.elastic_coeff_2 * strain_tensor[0][0] * minv3 * (1.0 + c.z);
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv2 * c.x;
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[0][1] * minv1 * c.y;
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv3 * c.y;
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[1][2] * minv2 * (1.0 + c.z);
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv3 * c.x;
		fz += -4.0 * elastic_param_c.elastic_coeff_3 * strain_tensor[2][0] * minv1 * (1.0 + c.z);

		// Compute total energy and assign one fouth on each particle
		Scalar e_total = elastic_param_c.elastic_coeff_1 * (
						  strain_tensor[0][0]*strain_tensor[0][0] 
						+ strain_tensor[1][1]*strain_tensor[1][1] 
						+ strain_tensor[2][2]*strain_tensor[2][2]
						)
				+ 2 * elastic_param_c.elastic_coeff_2 * (
						  strain_tensor[0][0] * strain_tensor[1][1]
						+ strain_tensor[1][1] * strain_tensor[2][2]
						+ strain_tensor[2][2] * strain_tensor[0][0]
						)
				+ 4 * elastic_param_c.elastic_coeff_3 * (
						  strain_tensor[0][1] * strain_tensor[0][1]
						+ strain_tensor[1][2] * strain_tensor[1][2]
						+ strain_tensor[2][0] * strain_tensor[2][0]
						);
							
		
		// Store the forces with w component storing energy of particle
		h_force.data[idx_0].x += fx.x;
		h_force.data[idx_0].y += fy.x;
		h_force.data[idx_0].z += fz.x;
		h_force.data[idx_0].w += e_total * 0.25;
		
		h_force.data[idx_1].x += fx.y;
		h_force.data[idx_1].y += fy.y;
		h_force.data[idx_1].z += fz.y;
		h_force.data[idx_1].w += e_total * 0.25;
		
		h_force.data[idx_2].x += fx.z;
		h_force.data[idx_2].y += fy.z;
		h_force.data[idx_2].z += fz.z;
		h_force.data[idx_2].w += e_total * 0.25;
		
		// force for fourth vertex follows N2L, negative of sum of other vertices.
		h_force.data[idx_3].x += -(fx.x + fx.y + fx.z);
		h_force.data[idx_3].y += -(fy.x + fy.y + fy.z);
		h_force.data[idx_3].z += -(fz.x + fz.y + fz.z);	
		h_force.data[idx_3].w += e_total * 0.25;
		
		}
    }

namespace detail
	{
	void export_Elastic(pybind11::module& m)
	{
	pybind11::class_<Elastic, ForceCompute, std::shared_ptr<Elastic>>(m, "Elastic")
	.def(pybind11::init<std::shared_ptr<SystemDefinition>,std::shared_ptr<TetrahedronData>,pybind11::array_t<Scalar>, pybind11::array_t<unsigned int>>())
	.def("setParams", &Elastic::setParamsPython)
	.def("getParams", &Elastic::getParams);
	}
	}

    } // namespace md
    } // namespace hoomd
	  //

