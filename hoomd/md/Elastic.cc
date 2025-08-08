#include <pybind11/pybind11.h>
#include <unordered_map>

#include "Elastic.h"

namespace hoomd{
		namespace md{
	

void Elastic::setReference(pybind11::array_t<Scalar> reference_positions, 
                                pybind11::array_t<unsigned int> reference_tags){
    //Validate that arrays from pybind are of correct shape
    auto reference_positions_indexer = reference_positions.unchecked<2>();
    auto reference_tags_indexer = reference_tags.unchecked<1>();

    const auto N_reference_pos = reference_positions_indexer.shape(0);

	const auto& box = m_pdata->getGlobalBox();

    if(reference_tags_indexer.shape(0) != reference_positions_indexer.shape(0)){
        throw std::invalid_argument("The array of tetrahedron vertex positions and tags must have the same number of rows (N_particles).");
    }
    if(reference_positions_indexer.shape(1) != 3){
        throw std::invalid_argument("The array of tetrahedron vertex positions must have three columns.");
    }
    //Initialize m_reference_vertex_displacements
    GPUArray<vec3<Scalar>> reference_vertex_displacements(3*m_tetrahedron_data->getN(),m_exec_conf);
    m_reference_vertex_displacements.swap(reference_vertex_displacements);
    ArrayHandle<vec3<Scalar>> h_reference_vertex_displacements(m_reference_vertex_displacements,
                                                    access_location::host,
                                                    access_mode::overwrite);

    //Initialize m_reference_inv_matrix
    GPUArray<vec3<Scalar>> reference_inv_matrix(3*m_tetrahedron_data->getN(), m_exec_conf);
    m_reference_inv_matrix.swap(reference_inv_matrix);
    ArrayHandle<vec3<Scalar>> h_reference_inv_matrix(m_reference_inv_matrix,
                                                    access_location::host,
                                                    access_mode::overwrite);

	// TODO: Populate m_reference_vertex_displacements and m_reference_inv_matrix
    const auto n_tetrahedra = static_cast<unsigned int>(m_tetrahedron_data->getN());
	
	std::unordered_map<unsigned int, unsigned int> index_map;

	for(unsigned int i=0; i < N_reference_pos; i++){
		index_map[reference_tags_indexer(i)] = i;
	}

	m_matrix_indexer = Index2D(n_tetrahedra,3);

	
	for(unsigned int i=0; i < n_tetrahedra; i++){
		const auto& tetrahedron = m_tetrahedron_data->getMembersByIndex(i);
		
		auto particle_index_0 = index_map[tetrahedron.tag[0]];
		auto particle_index_1 = index_map[tetrahedron.tag[1]];
		auto particle_index_2 = index_map[tetrahedron.tag[2]];
		auto particle_index_3 = index_map[tetrahedron.tag[3]];
	   	// TODO: come back late to add error checking

		auto r0 = vec3<Scalar>(reference_positions_indexer(particle_index_0,0),
                               reference_positions_indexer(particle_index_0,1),
                               reference_positions_indexer(particle_index_0,2));
		
		auto r1 = vec3<Scalar>(reference_positions_indexer(particle_index_1,0),
                               reference_positions_indexer(particle_index_1,1),
                               reference_positions_indexer(particle_index_1,2));
		
		auto r2 = vec3<Scalar>(reference_positions_indexer(particle_index_2,0),
                               reference_positions_indexer(particle_index_2,1),
                               reference_positions_indexer(particle_index_2,2));
		
		auto r3 = vec3<Scalar>(reference_positions_indexer(particle_index_3,0),
                               reference_positions_indexer(particle_index_3,1),
                               reference_positions_indexer(particle_index_3,2));

		h_reference_vertex_displacements.data[m_matrix_indexer(i,0)] = box.minImage(r0 - r3);
		h_reference_vertex_displacements.data[m_matrix_indexer(i,1)] = box.minImage(r1 - r3);
		h_reference_vertex_displacements.data[m_matrix_indexer(i,2)] = box.minImage(r2 - r3);

	}

}


void Elastic::computeForces(uint64_t timestep){
    // Create 
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_particle_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);

    // Zero data for force calculation
    m_force.zeroFill();
    m_virial.zeroFill();

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    const BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    ArrayHandle<typename TetrahedronData::members_t> h_tetrahedron(m_tetrahedron_data->getMembersArray(),
                                                   access_location::host,
                                                   access_mode::read);
    ArrayHandle<typeval_t> h_typeval(m_tetrahedron_data->getTypeValArray(),
                                     access_location::host,
                                     access_mode::read);

    ArrayHandle<vec3<Scalar>> h_reference_vertex_displacements(m_reference_vertex_displacements,
                                                access_location::host,
                                                access_mode::read);

    unsigned int max_local = m_pdata->getN() + m_pdata->getNGhosts();

    const size_t n_tetrahedra = m_tetrahedron_data->getN();
    
    // Calculate forces for each tetrahedron
	/*
    for(size_t i=0, i < n_tetrahedra, i++){
        const auto& tetrahedron = h_tetrahedon.data[i];
        unsigned int idx_i = h_particle_rtag.data[tetrahedron.tag[0]];
        unsigned int idx_j = h_particle_rtag.data[tetrahedron.tag[1]];
        unsigned int idx_k = h_particle_rtag.data[tetrahedron.tag[2]];
        unsigned int idx_l = h_particle_rtag.data[tetrahedron.tag[3]];
        //TODO: Modify this to get matrix elements
        unsigned int idx_ref_i = h_reference_rtag.data[tetrahedron.tag[0]];
        unsigned int idx_ref_j = h_reference_rtag.data[tetrahedron.tag[1]];
        unsigned int idx_ref_k = h_reference_rtag.data[tetrahedron.tag[2]];
        unsigned int idx_ref_l = h_reference_rtag.data[tetrahedron.tag[3]];

        if (idx_i >= max_local || idx_j >= max_local || idx_k >= max_local || idx_l >= max_local)
            {
            std::ostringstream stream;
            stream << "Error: tetrahedron " << tetrahedron.tag[0] << " " << tetrahedron.tag[1] << " " << tetrahedron.tag[2] << " " << tetrahedron.tag[3] << " is incomplete.";
            throw std::runtime_error(stream.str());
            }
        
        // Step 1: Get displacements
        vec3<Scalar> posi = vec3<Scalar>(h_pos.data[idx_i].x, h_pos.data[idx_i].y, h_pos.data[idx_i].z);
        vec3<Scalar> posj = vec3<Scalar>(h_pos.data[idx_j].x, h_pos.data[idx_j].y, h_pos.data[idx_j].z);
        vec3<Scalar> posk = vec3<Scalar>(h_pos.data[idx_k].x, h_pos.data[idx_k].y, h_pos.data[idx_k].z);
        vec3<Scalar> posl = vec3<Scalar>(h_pos.data[idx_l].x, h_pos.data[idx_l].y, h_pos.data[idx_l].z);

        vec3<Scalar> ref_posi = vec3<Scalar>(h_reference_postions.data[idx_ref_i].x, h_reference_postions.data[idx_ref_i].y, h_reference_postions.data[idx_ref_i].z);
        vec3<Scalar> ref_posj = vec3<Scalar>(h_reference_postions.data[idx_ref_j].x, h_reference_postions.data[idx_ref_j].y, h_reference_postions.data[idx_ref_j].z);
        vec3<Scalar> ref_posk = vec3<Scalar>(h_reference_postions.data[idx_ref_k].x, h_reference_postions.data[idx_ref_k].y, h_reference_postions.data[idx_ref_k].z);
        vec3<Scalar> ref_posl = vec3<Scalar>(h_reference_postions.data[idx_ref_l].x, h_reference_postions.data[idx_ref_l].y, h_reference_postions.data[idx_ref_l].z);
	

        // Step 2: Calculate Mapping coefficients (a, b, c)

        // Step 3: Calculate Strains (epsilons)

        // Step 4: Calculate Forces
	*/

    }

}

}
