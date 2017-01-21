# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Metal pair potentials.
"""

from hoomd.md import force;
from hoomd.md import nlist as nl # to avoid naming conflicts
import hoomd;
from hoomd import _hoomd
from hoomd.md import _md
from hoomd.metal import _metal

import math;
import sys;

class eam(force._force):
    R""" EAM pair potential.

    Args:
        file (str): Filename with potential tables in Alloy or FS format
        type (str): Type of file potential ('Alloy', 'FS')
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list (default of None automatically creates a global cell-list based neighbor list)
        ifinter (bool): Interpolation of the array turn on (True) or off (False)
        nrho (int):  (if ifinter=True) the number of values in the interpolated embedding function arrays
        nr (int): (if ifinter=True) the number of values in the interpolated density function and the pair potential function arrays

    :py:class:`eam` specifies that a EAM (embedded atom method) pair potential should be applied between every
    non-excluded particle pair in the simulation.

    No coefficients need to be set for :py:class:`eam`. All specifications, including the cutoff radius, form of the
    potential, etc. are read in from the specified file.

    Denser or sparser potential data points can be used by turning on the interpolation function, which adopts Stephen Foiles's
    interpolation method in LAMMPS to interpolate the original potential data, and export a file "eambyhoomd.pot" in the current
    directory. The number of values in the interpolated embedding function arrays is set by nrho, that in the interpolated density
    function and the pair potential function is set by nr. It is not suggested to set nrho or nr above 10000, in practice, more than
    10000 data points will not improve the accuracy very much **ADD REF**, but will saturate the memory easily.

    Particle type names must match those referenced in the EAM potential file.

    Two file formats are supported: *Alloy* and *FS*. They are described in LAMMPS documentation
    (commands eam/alloy and eam/fs) here: http://lammps.sandia.gov/doc/pair_eam.html
    and are also described here: http://enpub.fulton.asu.edu/cms/potentials/submain/format.htm

    .. attention::
        EAM is **NOT** supported in MPI parallel simulations.

    .. danger::
        HOOMD-blue's EAM implementation is known to be broken.

    Example::

        nl = nlist.cell()
        eam = pair.eam(file='al1.mendelev.eam.fs', type='FS', nlist=nl)
        eam = pair.eam(file='al1.mendelev.eam.alloy', type='Alloy', nlist=nl, ifinter=True, nrho=10000, nr=10000)

    """
    def __init__(self, file, type, nlist, ifinter=False, nrho=10000, nr=10000):
        c = hoomd.cite.article(cite_key = 'morozov2011',
                         author=['I V Morozov','A M Kazennova','R G Bystryia','G E Normana','V V Pisareva','V V Stegailova'],
                         title = 'Molecular dynamics simulations of the relaxation processes in the condensed matter on GPUs',
                         journal = 'Computer Physics Communications',
                         volume = 182,
                         number = 9,
                         pages = '1974--1978',
                         year = '2011',
                         doi = '10.1016/j.cpc.2010.12.026',
                         feature = 'EAM')
        hoomd.cite._ensure_global_bib().add(c)

        hoomd.util.print_status_line();

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("pair.eam is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up pair potential.")

        # initialize the base class
        force._force.__init__(self);
        # Translate type
        if(type == 'Alloy'): type_of_file = 0;
        elif(type == 'FS'): type_of_file = 1;
        else: raise RuntimeError('Unknown EAM input file type');
        # Translate interpolation command
        if(ifinter == True): inter = 1;
        elif(ifinter == False): inter = 0;
        else: raise RuntimeError('Unknown EAM interpolation command');

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _metal.EAMForceCompute(hoomd.context.current.system_definition, file, type_of_file, inter, nrho, nr);
        else:
            self.cpp_force = _metal.EAMForceComputeGPU(hoomd.context.current.system_definition, file, type_of_file, inter, nrho, nr);

        #After load EAMForceCompute we know r_cut from EAM potential`s file. We need update neighbor list.
        self.r_cut_new = self.cpp_force.get_r_cut();
        self.nlist = nlist
        self.nlist.subscribe(lambda : self.get_rcut())
        self.nlist.update_rcut()

        #Load neighbor list to compute.
        self.cpp_force.set_neighbor_list(self.nlist.cpp_nlist);
        if hoomd.context.exec_conf.isCUDAEnabled():
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full);

        hoomd.context.msg.notice(2, "Set r_cut = " + str(self.r_cut_new) + " from potential`s file '" +  str(file) + "'.\n");

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);
        self.pair_coeff = hoomd.md.pair.coeff();

    def get_rcut(self):
        # go through the list of only the active particle types in the simulation
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0, ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))
        # update the rcut by pair type
        r_cut_dict = nl.rcut()
        for i in range(0, ntypes):
            for j in range(i, ntypes):
                # get the r_cut value
                r_cut_dict.set_pair(type_list[i], type_list[j], self.r_cut_new)
        return r_cut_dict

    def update_coeffs(self):
        # check that the pair coefficients are valid
        pass;
