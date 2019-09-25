# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: jglaser / All Developers are free to add commands for new features

""" MPI communication interface

Use methods in this module to query the number of MPI ranks, the current rank, etc...
"""

from hoomd import _hoomd
import hoomd;

import sys;


class decomposition(object):
    """ Set the domain decomposition.

    Args:
        x (list): First nx-1 fractional domain widths (if *nx* is None)
        y (list): First ny-1 fractional domain widths (if *ny* is None)
        z (list): First nz-1 fractional domain widths (if *nz* is None)
        nx (int): Number of processors to uniformly space in x dimension (if *x* is None)
        ny (int): Number of processors to uniformly space in y dimension (if *y* is None)
        nz (int): Number of processors to uniformly space in z dimension (if *z* is None)
        linear (bool): (MPI only) Force a slab (1D) decomposition along the z-direction
        onelevel (bool): (MPI only) Disable node-local (two-level) domain decomposition

    A single domain decomposition is defined for the simulation.
    A standard domain decomposition divides the simulation box into equal volumes along the Cartesian axes while minimizing
    the surface area between domains. This works well for systems where particles are uniformly distributed and
    there is equal computational load for each domain, and is the default behavior in HOOMD-blue. If no decomposition is
    specified for an MPI run, a uniform decomposition is automatically constructed on initialization.

    In simulations with density gradients, such as a vapor-liquid interface, there can be a considerable imbalance of
    particles between different ranks. The simulation time then becomes limited by the slowest processor. It may then be
    advantageous in certain systems to create domains of unequal volume, for example, by increasing the volume of less
    dense regions of the simulation box in order to balance the number of particles.

    The decomposition command allows the user to control the geometry and positions of the decomposition.
    The fractional width of the first :math:`n_i - 1` domains is specified along each dimension, where
    :math:`n_i` is the number of ranks desired along dimension :math:`i`. If no cut planes are specified, then a uniform
    spacing is assumed. The number of domains with uniform spacing can also be specified. If the desired decomposition
    is not commensurate with the number of ranks available (for example, a 3x3x3 decomposition when only 8 ranks are
    available), then a default uniform spacing is chosen. For the best control, the user should specify the number of
    ranks in each dimension even if uniform spacing is desired.

    decomposition can only be called *before* the system is initialized, at which point the particles are decomposed.
    An error is raised if the system is already initialized.

    The decomposition can be adjusted dynamically if the best static decomposition is not known, or the system
    composition is changing dynamically. For this associated command, see update.balance().

    Priority is always given to specified arguments over the command line arguments. If one of these is not set but
    a command line option is, then the command line option is used. Otherwise, a default decomposition is chosen.

    Examples::

        comm.decomposition(x=0.4, ny=2, nz=2)
        comm.decomposition(nx=2, y=0.8, z=[0.2,0.3])

    Warning:
        The decomposition command will override specified command line options.

    Warning:
        This command must be invoked *before* the system is initialized because particles are decomposed at this time.

    Note:
        The domain size cannot be chosen arbitrarily small. There are restrictions placed on the decomposition by the
        ghost layer width set by the pair potentials. An error will be raised at run time if the ghost layer width
        exceeds half the shortest domain size.

    Warning:
        Both fractional widths and the number of processors cannot be set simultaneously, and an error will be
        raised if both are set.
    """

    def __init__(self, x=None, y=None, z=None, nx=None, ny=None, nz=None, linear=False, onelevel=False):

        # check that the context has been initialized though
        if hoomd.context.current is None:
            raise RuntimeError("Cannot initialize decomposition without context.initialize() first")

        # check that system is not initialized
        if hoomd.context.current.system is not None:
            hoomd.context.current.device.cpp_msg.error("comm.decomposition: cannot modify decomposition after system is initialized. Call before init.*\n")
            raise RuntimeError("Cannot create decomposition after system is initialized. Call before init.*")

        # make sure MPI is enabled if any arguments are not None
        if (x or y or z or nx or ny or nz) and (not _hoomd.is_MPI_available()):
            raise RuntimeError("the x, y, z, nx, ny, nz options are only available in MPI builds")

        self._onelevel = onelevel  # cache this for later when we can make the cpp object

        # check that there are ranks available for decomposition
        if hoomd.context.current.device.comm.cpp_mpi_conf == 1:
            hoomd.context.current.device.cpp_msg.warning("Only 1 rank in system, ignoring decomposition to use optimized code pathways.\n")
            return
        else:
            self.x = []
            self.y = []
            self.z = []
            self.nx = 0
            self.ny = 0
            self.nz = 0
            self.uniform_x = True
            self.uniform_y = True
            self.uniform_z = True

            self.set_params(x,y,z,nx,ny,nz)

            """
            # do a one time update of the cuts to the global values if a global is set
            if not self.x and self.nx == 0 and hoomd.context.options.nx is not None:
                self.nx = hoomd.context.options.nx
                self.uniform_x = True
            if not self.y and self.ny == 0 and hoomd.context.options.ny is not None:
                self.ny = hoomd.context.options.ny
                self.uniform_y = True
            if not self.z and self.nz == 0:
                if linear:
                    self.nz = hoomd.context.current.device.cpp_mpi_conf.getNRanks()
                    self.uniform_z = True
                elif hoomd.context.options.nz is not None:
                    self.nz = hoomd.context.options.nz
                    self.uniform_z = True
            """

            # set the global decomposition to this class
            if hoomd.context.current.decomposition is not None:
                hoomd.context.current.device.cpp_msg.warning("comm.decomposition: overriding currently defined domain decomposition\n")

            hoomd.context.current.decomposition = self

    def set_params(self,x=None,y=None,z=None,nx=None,ny=None,nz=None):
        """Set parameters for the decomposition before initialization.

        Args:
            x (list): First nx-1 fractional domain widths (if *nx* is None)
            y (list): First ny-1 fractional domain widths (if *ny* is None)
            z (list): First nz-1 fractional domain widths (if *nz* is None)
            nx (int): Number of processors to uniformly space in x dimension (if *x* is None)
            ny (int): Number of processors to uniformly space in y dimension (if *y* is None)
            nz (int): Number of processors to uniformly space in z dimension (if *z* is None)

        Examples::

            decomposition.set_params(x=[0.2])
            decomposition.set_params(nx=1, y=[0.3,0.4], nz=2)
        """

        if (x is not None and nx is not None) or (y is not None and ny is not None) or (z is not None and nz is not None):
            hoomd.context.current.device.cpp_msg.error("comm.decomposition: cannot set fractions and number of processors simultaneously\n")
            raise RuntimeError("Cannot set fractions and number of processors simultaneously")

        # if x is set, use it. otherwise, if nx is set, compute x and set it
        if x is not None:
            # recast single floats as lists that can be iterated, this is the only single input we should expect
            if isinstance(x, float):
                self.x = [x]
            else:
                self.x = x
            self.uniform_x = False
        elif nx is not None:
            self.nx = nx
            self.uniform_x = True

        # do the same in y
        if y is not None:
            if isinstance(y, float):
                self.y = [y]
            else:
                self.y = y
            self.uniform_y = False
        elif ny is not None:
            self.ny = ny
            self.uniform_y = True

        # do the same in z (but also use the linear command line option if it is present, which supersedes nz)
        if z is not None:
            if isinstance(z, float):
                self.z = [z]
            else:
                self.z = z
            self.uniform_z = False
        elif nz is not None:
            self.nz = nz
            self.uniform_z = True

    ## \internal
    # \brief Delayed construction of the C++ object for this balanced decomposition
    # \param box Global simulation box for decomposition
    def _make_cpp_decomposition(self, box):
        # if the box is uniform in all directions, just use these values
        if self.uniform_x and self.uniform_y and self.uniform_z:
            self.cpp_dd = _hoomd.DomainDecomposition(hoomd.context.current.device.cpp_exec_conf, box.getL(), self.nx, self.ny, self.nz, not self._onelevel)
            return self.cpp_dd

        # otherwise, make the fractional decomposition
        try:
            fxs = _hoomd.std_vector_scalar()
            fys = _hoomd.std_vector_scalar()
            fzs = _hoomd.std_vector_scalar()

            # if uniform, correct the fractions to be uniform as well
            if self.uniform_x and self.nx > 0:
                self.x = [1.0/self.nx]*(self.nx-1)
            if self.uniform_y and self.ny > 0:
                self.y = [1.0/self.ny]*(self.ny-1)
            if self.uniform_z and self.nz > 0:
                self.z = [1.0/self.nz]*(self.nz-1)

            sum_x = sum_y = sum_z = 0.0
            tol = 1.0e-5
            for i in self.x:
                if i <= -tol or i >= 1.0 - tol:
                    hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fxs.append(i)
                sum_x += i
            if sum_x >= 1.0 - tol or sum_x <= -tol:
                hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in x must lie between 0.0 and 1.0")

            for i in self.y:
                if i <= -tol or i >= 1.0 - tol:
                    hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fys.append(i)
                sum_y += i
            if sum_y >= 1.0 - tol or sum_y <= -tol:
                hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in y must lie between 0.0 and 1.0")

            for i in self.z:
                if i <= -tol or i >= 1.0 - tol:
                    hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fzs.append(i)
                sum_z += i
            if sum_z >= 1.0 - tol or sum_z <= -tol:
                hoomd.context.current.device.cpp_msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in z must lie between 0.0 and 1.0")

            self.cpp_dd = _hoomd.DomainDecomposition(hoomd.context.current.device.cpp_exec_conf, box.getL(), fxs, fys, fzs)
            return self.cpp_dd

        except TypeError as te:
            hoomd.context.current.device.cpp_msg.error("Fractional cuts must be iterable (list, tuple, etc.)\n")
            raise te

class Communicator(object):
    """
    MPI communicator

    Args:
        mpi_comm: Accepts an mpi4py communicator. Use this argument to perform many independent hoomd simulations
                where you communicate between those simulations using your own mpi4py code.
        nrank (int): (MPI) Number of ranks to include in a partition
    """

    def __init__(self, mpi_comm=None, nrank=None):

        # check nrank
        if nrank is not None:
            if not _hoomd.is_MPI_available():
                raise RuntimeError("The nrank option is only available in MPI builds.\n")

        mpi_available = _hoomd.is_MPI_available();

        self.cpp_mpi_conf = None

        # create the specified configuration
        if mpi_comm is None:
            self.cpp_mpi_conf = _hoomd.MPIConfiguration();
        else:
            if not mpi_available:
                raise RuntimeError("mpi_comm is not supported in serial builds");

            handled = False;

            # pass in pointer to MPI_Comm object provided by mpi4py
            try:
                import mpi4py
                if isinstance(mpi_comm, mpi4py.MPI.Comm):
                    addr = mpi4py.MPI._addressof(mpi_comm);
                    self.cpp_mpi_conf = _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(addr);
                    handled = True
            except ImportError:
                # silently ignore when mpi4py is missing
                pass

            # undocumented case: handle plain integers as pointers to MPI_Comm objects
            if not handled and isinstance(mpi_comm, int):
                self.cpp_mpi_conf = _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(mpi_comm);
                handled = True

            if not handled:
                raise RuntimeError("Invalid mpi_comm object: {}".format(mpi_comm));

        if nrank is not None:
            # check validity
            if (self.cpp_mpi_conf.getNRanksGlobal() % nrank):
                raise RuntimeError('Total number of ranks is not a multiple of --nrank');

            # split the communicator into partitions
            self.cpp_mpi_conf.splitPartitions(nrank)

    @property
    def num_ranks(self):
        """ Get the number of ranks in this partition.

        Returns:
            The number of MPI ranks in this partition.

        Note:
            Returns 1 in non-mpi builds.
        """

        hoomd.context._verify_init();
        if _hoomd.is_MPI_available():
            return self.cpp_mpi_conf.getNRanks();
        else:
            return 1;

    @property
    def rank(self):
        """ Get the current rank.

        Returns:
            Index of the current rank in this partition.

        Note:
            Always returns 0 in non-mpi builds.
        """

        hoomd.context._verify_init();

        if _hoomd.is_MPI_available():
            return self.cpp_mpi_conf.getRank()
        else:
            return 0;

    @property
    def partition(self):
        """ Get the current partition index.

        Returns:
            Index of the current partition.

        Note:
            Always returns 0 in non-mpi builds.
        """
        hoomd.context._verify_init();

        if _hoomd.is_MPI_available():
            return self.cpp_mpi_conf.getPartition()
        else:
            return 0;

    def barrier_all(self):
        """ Perform a MPI barrier synchronization across the whole MPI run.

        Note:
            Does nothing in in non-MPI builds.
        """
        if _hoomd.is_MPI_available():
            _hoomd.mpi_barrier_world();

    def barrier(self):
        """ Perform a MPI barrier synchronization across all ranks in the partition.

        Note:
            Does nothing in in non-MPI builds.
        """
        hoomd.context._verify_init();

        if _hoomd.is_MPI_available():
            self.cpp_mpi_conf.barrier()

