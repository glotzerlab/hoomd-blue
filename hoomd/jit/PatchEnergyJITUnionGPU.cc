#ifdef ENABLE_HIP
#include "PatchEnergyJITUnionGPU.h"

#include "hoomd/jit/EvaluatorUnionGPU.cuh"

//! Set the per-type constituent particles
void PatchEnergyJITUnionGPU::setParam(unsigned int type,
    pybind11::list types,
    pybind11::list positions,
    pybind11::list orientations,
    pybind11::list diameters,
    pybind11::list charges,
    unsigned int leaf_capacity)
    {
    // set parameters in base class
    PatchEnergyJITUnion::setParam(type, types, positions, orientations, diameters, charges, leaf_capacity);

    unsigned int N = len(positions);

    hpmc::detail::OBB *obbs = new hpmc::detail::OBB[N];

    jit::union_params_t params(N, true);

    // set shape parameters
    for (unsigned int i = 0; i < N; i++)
        {
        pybind11::list positions_i = pybind11::cast<pybind11::list>(positions[i]);
        vec3<float> pos = vec3<float>(pybind11::cast<float>(positions_i[0]), pybind11::cast<float>(positions_i[1]), pybind11::cast<float>(positions_i[2]));
        pybind11::list orientations_i = pybind11::cast<pybind11::list>(orientations[i]);
        float s = pybind11::cast<float>(orientations_i[0]);
        float x = pybind11::cast<float>(orientations_i[1]);
        float y = pybind11::cast<float>(orientations_i[2]);
        float z = pybind11::cast<float>(orientations_i[3]);
        quat<float> orientation(s, vec3<float>(x,y,z));

        float diameter = pybind11::cast<float>(diameters[i]);
        float charge = pybind11::cast<float>(charges[i]);
        params.mtype[i] = pybind11::cast<unsigned int>(types[i]);
        params.mpos[i] = pos;
        params.morientation[i] = orientation;
        params.mdiameter[i] = diameter;
        params.mcharge[i] = charge;

        // use a point-sized OBB
        obbs[i] = hpmc::detail::OBB(pos,0.0);

        // we do not support exclusions
        obbs[i].mask = 1;
        }

    // build tree and store proxy structure
    hpmc::detail::OBBTree tree;
    bool internal_nodes_spheres = false;
    tree.buildTree(obbs, N, leaf_capacity, internal_nodes_spheres);
    delete [] obbs;
    bool managed = true;
    params.tree = hpmc::detail::GPUTree(tree, managed);

    // store result
    m_d_union_params[type] = params;

    // cudaMemadviseReadMostly
    m_d_union_params[type].set_memory_hint();

    m_params_updated = true;
    }

void export_PatchEnergyJITUnionGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITUnionGPU, PatchEnergyJITUnion, std::shared_ptr<PatchEnergyJITUnionGPU> >(m, "PatchEnergyJITUnionGPU")
            .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                                 std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, const std::string&,
                                 const std::string&, const std::string&,
                                 const std::string&,
                                 unsigned int>())
            .def("setParam",&PatchEnergyJITUnionGPU::setParam)
            ;
    }
#endif
