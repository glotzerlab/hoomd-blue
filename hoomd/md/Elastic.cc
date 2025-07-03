#include Elastic.h

virtual void setReference(pybind11::array_t<Scalar> reference_positions, 
                                pybind11::array_t<size_t> reference_tags){
    //Validate that arrays from pybind are of correct shape
    auto reference_positions_indexer = reference_positions.unchecked<2>();
    auto reference_tags_indexer = reference_tags.unchecked<1>();

    const N_reference_pos = reference_positions_indexer.shape(0);

    if(reference_tags_indexer.shape(0) != reference_positions_indexer.shape(0)){
        throw std::invalid_argument("The array of tetrahedron vertex positions and tags must have the same number of rows (N_particles).");
    }
    if(reference_positions_indexer.shape(1) != 3){
        throw std::invalid_argument("The array of tetrahedron vertex positions must have three columns.");
    }

    GPUArray<vec3<Scalar>> reference_postions(reference_positions_indexer.shape(0));
    GPUArray<uint32_t> m_reference_rtag(m_pdata->getNParticles());

    m_reference_postions.swap(reference_postions);
    m_reference_tags.swap(reference_tags);

    ArrayHandle<vec3<Scalar>> h_reference_postions(m_reference_postions,
                                                    access_location::host,
                                                    access_mode::overwrite);

    ArrayHandle<uint32_t> h_reference_tags(m_reference_tags,
                                                    access_location::host,
                                                    access_mode::overwrite);

    for(size_t i=0; i < h_reference_tags.getNumElements(); i++){
        h_reference_tags.data[i] = UINT32_MAX;
    }

    for(size_t i=0; i < N_reference_pos; i++){
        h_reference_tags.data[reference_tags_indexer(i)] = i;
        h_reference_positions.data[i] = vec3<Scalar>(reference_positions_indexer(i,0),
                                                        reference_positions_indexer(i,1),
                                                        reference_positions_indexer(i,2));
    }

    
}