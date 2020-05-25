#ifdef ENABLE_HIP
#include "PatchEnergyJITUnionGPU.h"

#include "hoomd/jit/EvaluatorUnionGPU.cuh"
#include <pybind11/stl.h>

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

        // use a spherical OBB of radius 0.5*d
        obbs[i] = hpmc::detail::OBB(pos,0.5f*diameter);

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
    }

//! Kernel driver for kernel::hpmc_narrow_phase_patch
void PatchEnergyJITUnionGPU::computePatchEnergyGPU(const gpu_args_t& args, hipStream_t hStream)
    {
    #ifdef __HIP_PLATFORM_NVCC__
    assert(args.d_postype);
    assert(args.d_orientation);

    unsigned int param = m_tuner_narrow_patch->getParam();
    unsigned int block_size = param/1000000;
    unsigned int req_tpp = (param%1000000)/100;
    unsigned int eval_threads = param % 100;

    this->m_exec_conf->beginMultiGPU();
    m_tuner_narrow_patch->begin();

    // choose a block size based on the max block size by regs (max_block_size) and include dynamic shared memory usage
    unsigned int run_block_size = std::min(block_size, m_gpu_factory.getKernelMaxThreads(0, eval_threads, block_size)); // fixme GPU 0

    unsigned int tpp = std::min(req_tpp,run_block_size);
    while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
        {
        tpp--;
        }

    unsigned int n_groups = run_block_size/(tpp*eval_threads);

    // truncate blockDim.z
    auto& devprop = m_exec_conf->dev_prop;
    n_groups = std::min((unsigned int) devprop.maxThreadsDim[2], n_groups);
    unsigned int max_queue_size = n_groups*tpp;

    const unsigned int min_shared_bytes = args.num_types * sizeof(Scalar) +
                                          m_d_union_params.size()*sizeof(jit::union_params_t);

    unsigned int shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
        + max_queue_size * 2 * sizeof(unsigned int)
        + min_shared_bytes;

    if (min_shared_bytes >= devprop.sharedMemPerBlock)
        throw std::runtime_error("Insufficient shared memory for HPMC kernel: reduce number of particle types or size of shape parameters");

    unsigned int kernel_shared_bytes = m_gpu_factory.getKernelSharedSize(0, eval_threads, block_size); //fixme GPU 0
    while (shared_bytes + kernel_shared_bytes >= devprop.sharedMemPerBlock)
        {
        run_block_size -= devprop.warpSize;
        if (run_block_size == 0)
            throw std::runtime_error("Insufficient shared memory for HPMC kernel");

        tpp = std::min(req_tpp, run_block_size);
        while (eval_threads*tpp > run_block_size || run_block_size % (eval_threads*tpp) != 0)
            {
            tpp--;
            }

        n_groups = run_block_size / (tpp*eval_threads);

        // truncate blockDim.z
        n_groups = std::min((unsigned int)devprop.maxThreadsDim[2], n_groups);

        max_queue_size = n_groups*tpp;

        shared_bytes = n_groups * (4*sizeof(unsigned int) + 2*sizeof(Scalar4) + 2*sizeof(Scalar3) + 2*sizeof(Scalar))
            + max_queue_size * 2 * sizeof(unsigned int)
            + min_shared_bytes;
        }

    // allocate some extra shared mem to store union shape parameters
    unsigned int max_extra_bytes = m_exec_conf->dev_prop.sharedMemPerBlock - shared_bytes - kernel_shared_bytes;

    // determine dynamically requested shared memory
    char *ptr = (char *)nullptr;
    unsigned int available_bytes = max_extra_bytes;
    for (unsigned int i = 0; i < m_d_union_params.size(); ++i)
        {
        m_d_union_params[i].allocate_shared(ptr, available_bytes);
        }
    unsigned int extra_bytes = max_extra_bytes - available_bytes;
    shared_bytes += extra_bytes;

    dim3 thread(eval_threads, tpp, n_groups);

    auto& gpu_partition = args.gpu_partition;

    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        const unsigned int num_blocks = (nwork + n_groups - 1)/n_groups;

        dim3 grid(num_blocks, 1, 1);

        unsigned int N_old = args.N + args.N_ghost;

        // configure the kernel
        auto launcher = m_gpu_factory.configureKernel(idev, grid, thread, shared_bytes, hStream, eval_threads, block_size);

        CUresult res = launcher(args.d_postype,
            args.d_orientation,
            args.d_trial_postype,
            args.d_trial_orientation,
            args.d_charge,
            args.d_diameter,
            args.d_excell_idx,
            args.d_excell_size,
            args.excli,
            args.d_nlist_old,
            args.d_energy_old,
            args.d_nneigh_old,
            args.d_nlist_new,
            args.d_energy_new,
            args.d_nneigh_new,
            args.maxn,
            args.num_types,
            args.box,
            args.ghost_width,
            args.cell_dim,
            args.ci,
            N_old,
            args.N,
            args.r_cut_patch,
            args.d_additive_cutoff,
            args.d_overflow,
            max_queue_size,
            range.first,
            nwork,
            max_extra_bytes);

        if (res != CUDA_SUCCESS)
            {
            char *error;
            cuGetErrorString(res, const_cast<const char **>(&error));
            throw std::runtime_error("Error launching NVRTC kernel: "+std::string(error));
            }
        }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_narrow_patch->end();
    m_exec_conf->endMultiGPU();
    #endif
    }

void export_PatchEnergyJITUnionGPU(pybind11::module &m)
    {
    pybind11::class_<PatchEnergyJITUnionGPU, PatchEnergyJITUnion, std::shared_ptr<PatchEnergyJITUnionGPU> >(m, "PatchEnergyJITUnionGPU")
            .def(pybind11::init< std::shared_ptr<SystemDefinition>,
                                 std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, Scalar, const unsigned int,
                                 const std::string&, const std::string&,
                                 const std::vector<std::string>&,
                                 const std::string&,
                                 unsigned int>())
            .def("setParam",&PatchEnergyJITUnionGPU::setParam)
            ;
    }
#endif
