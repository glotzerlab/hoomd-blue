.. _mpi-domain-decomposition:

MPI domain decomposition
========================

Overview
--------

HOOMD-blue supports multi-GPU (and multi-CPU) simulations using MPI. It uses
a spatial domain decomposition approach similar to the one used by LAMMPS.
Every GPU is assigned a sub-domain of the simulation box, the dimensions of which
are calculated by dividing the lengths of the simulation box by the
number of processors per dimension. These domain boundaries can also be adjusted
to different fractional widths while still maintaining a 3d grid, which can be
advantageous in systems with density gradients. The product of the number of processors along
all dimensions must equal the number of processors in the MPI job. As in single-GPU
simulations, there is a one-to-one mapping between host CPU cores (MPI ranks)
and the GPUs.

Job scripts do not need to be modified to take advantage of multi-GPU execution. However,
not all features are available in MPI mode. The list of single-GPU only features can be found below.

See `J. Glaser et. al. 2015 <http://dx.doi.org/10.1016/j.cpc.2015.02.028>`_ for more implementation details.

Compilation
-----------

For detailed compilation instructions, see :ref:`compile-hoomd`.

Compilation flags pertinent to MPI simulations are:

- **ENABLE_MPI** (to enable multi-GPU simulations, must be set to \b ON)
- **ENABLE_MPI_CUDA** (optional, enables CUDA-aware MPI library support, see below)

Usage
-----

To execute a hoomd job script on multiple GPUs, run::

    mpirun -n 8 python script.py --mode=gpu

This will execute HOOMD on 8 processors. HOOMD automatically detects which GPUs are available and assigns them to MPI
ranks.  The syntax and name of the ``mpirun`` command may be different
between different MPI libraries and system architectures, check with your system documentation to find out what
launcher to use. When running on multiple nodes, the job script must be available to all nodes via a network file system.

HOOMD chooses the best spatial sub-division according to a minimum-area rule. If needed, the dimensions of the
decomposition be specified using the
**linear**, **nx**, **ny** and **nz** :ref:`command-line-options`.
If your intention is to run HOOMD on a single GPU, you can
invoke HOOMD with no MPI launcher::

    python script.py

instead of giving the ``-n 1`` argument to ``mpirun``.

.. warning:: Some cluster environments do not allow this and require the MPI launcher be used even for single rank jobs.

HOOMD-blue can also execute on many **CPU cores** in parallel::

    mpirun -n 16 python script.py --mode=cpu

GPU selection in MPI runs
-------------------------

HOOMD-blue uses information from ``mpirun`` to determine the *local rank* on a node (0,1,2,...). Each rank will use the
GPU id matching the local rank modulus the number of GPUs on the node. In this mode, do not run more ranks per node than
there are GPUs or you will oversubscribe the GPUs. This selection mechanism selects GPUs from within the set of GPUs
provided by the cluster scheduler.

In some MPI stacks, such as Intel MPI, this information is unavailable and HOOMD falls back on selecting
``gpu_id = global_rank % num_gpus_on_node`` and issues a notice message.
This mode only works on clusters where scheduling is performed by node (not by core) and there are a uniform
number of GPUs on each node.

In any case, a status message is printed on startup that lists which ranks are using which GPU ids. You can use this
to verify proper GPU selection.

Best practices
--------------

HOOMD-blue's multi-GPU performance depends on many factors, such as the model of the
actual GPU used, the type of interconnect between nodes, whether the MPI library
supports CUDA, etc. Below we list some recommendations for obtaining optimal
performance.

System size
^^^^^^^^^^^

Performance depends greatly on system size. Runs with fewer particles per GPU will execute slower due to communications
overhead. HOOMD-blue has decent strong scaling down to small numbers of particles per GPU, but to obtain high efficiency
(more than 60%) typical benchmarks need 100,000 or more particles per GPU. You should benchmark your own system of
interest with short runs to determine a reasonable range of efficient scaling behavior. Different potentials and/or
cutoff radii can greatly change scaling behavior.

CUDA-aware MPI libraries
^^^^^^^^^^^^^^^^^^^^^^^^

The main benefit of using a CUDA-enabled MPI library is that it enables intra-node
peer-to-peer (P2P) access between several GPUs on the same PCIe root complex, which increases
bandwidth. Secondarily, it may offer some additional optimization
for direct data transfer between the GPU and a network adapter.
To use these features with an MPI library that supports it,
set ``ENABLE_MPI_CUDA`` to **ON** for compilation.

Currently, we recommend building with ``ENABLE_MPI_CUDA`` **OFF**. On MPI libraries available at time of release,
enabling ``ENABLE_MPI_CUDA`` cuts performance in half. Systems with *GPUDirect RDMA* enabled improve on this somewhat,
but even on such systems typical benchmarks still run faster with ``ENABLE_MPI_CUDA`` **OFF**.

GPUDirect RDMA
^^^^^^^^^^^^^^

HOOMD does support *GPUDirect RDMA* with network adapters that support it (i.e. Mellanox) and compatible GPUs (Kepler),
through a CUDA-aware MPI library (i.e. OpenMPI 1.7.5 or MVAPICH 2.0b GDR). On HOOMD's side, nothing is required beyond setting
``ENABLE_MPI_CUDA`` to **ON** before compilation. On the side of the MPI library, special flags may need to be set
to enable *GPUDirect RDMA*, consult the documentation of your MPI library for that.

Slab decomposition
^^^^^^^^^^^^^^^^^^

For small numbers of GPUs per job (typically <= 8 for cubic boxes) that are non-prime,
the performance may be increased by using a slab decomposition.
A one-dimensional decomposition is enforced if the ``--linear``
command line option (:ref:`command-line-options`) is given.

Neighbor list buffer length (r_buff)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimum value of the **r_buff** value of the neighbor list
may be different between single- and multi-GPU runs. In multi-GPU
runs, the buffering length also determines the width of the ghost layer
runs and sets the size of messages exchanged between the processors.
To determine the optimum value, use :py:meth:`hoomd.md.nlist.nlist.tune()`.
command with the same number of MPI ranks that will be used for the production simulation.

Running with multiple partitions (--nrank command-line option)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HOOMD-blue supports simulation of multiple independent replicas, with the same
number of GPUs per replica. To enable multi-replica mode, and to partition
the total number of ranks **N** into **p = N/n** replicas, where **n** is the
number of GPUs per replica, invoke HOOMD-blue with the **--nrank=n**
command line option (see :ref:`command-line-options`).

Inside the command script, the current partition can be queried using
:py:func:`hoomd.comm.get_partition()`.

Dynamic load balancing
----------------------

HOOMD-blue supports non-uniform domain decomposition for systems with density gradients. A static domain decomposition
on a regular 3d grid but non-uniform widths can be constructed using
:py:class:`hoomd.comm.decomposition`. Here, either the number of processors in a uniform
decomposition or the fractional widths of :math:`n-1` domains can be set. Dynamic load balancing can be applied
to any domain decomposition either one time or periodically throughout the simulation using
:py:class:`hoomd.update.balance`. The domain boundaries are adjusted to attempt to place an
equal number of particles on each rank. The overhead from periodically updating the domain boundaries is reasonably
small, so most simulations with non-uniform particle distributions will benefit from periodic dynamic load balancing.

Troubleshooting
^^^^^^^^^^^^^^^

- **My simulation does not run significantly faster on exactly two GPUs compared to one GPU.**
   This is expected.  HOOMD uses special optimizations for single-GPU runs, which
   means that there is no overhead due to MPI calls. The communication overhead can be
   20-25\% of the total performance, and is only incurred when running on more
   than one GPU.

- **I get a message saying "Bond incomplete"**
   In multi-GPU simulations, there is an implicit restriction on the maximal length of a single
   bond. A bond cannot be longer than half the local domain size. If this happens,
   an error is thrown. The problem can be fixed by running HOOMD on fewer
   processors, or with a larger box size.

- **Simulations with large numbers of nodes are slow.**
   In simulations involving many nodes, collective MPI calls can take a significant portion
   of the run time. To find out if these are limiting you, run the simulation with
   the ``profile=True`` option to the :py:func:`hoomd.run()` command.
   One reason for slow performance can be the distance check, which, by default,
   is applied every step to check if the neighbor list needs to be rebuild. It requires
   synchronization between all MPI ranks and is therefore slow.
   See :py:meth:`hoomd.md.nlist.nlist.set_params()` to increase the interval (**check_period**)
   between distance checks, to improve performance.

- **My simulation crashes on multiple GPUs when I set ENABLE_MPI_CUDA=ON**
   First, check that cuda-aware MPI support is enabled in your MPI library. Usually this is
   determined at compile time of the MPI library. For MVAPICH2, HOOMD automatically sets
   the required environment variable **MV2_USE_CUDA=1**. If you are using *GPUDirect RDMA*
   in a dual-rail configuration,  special considerations need to be taken to ensure correct GPU-core affinity,
   not doing so may result in crashing or slow simulations.
