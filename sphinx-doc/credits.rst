Credits
=======

HOOMD-blue Developers
---------------------

The following people contributed to the :py:mod:`hoomd` and :py:mod:`hoomd.md` packages.

Joshua Anderson, University of Michigan - **Lead developer**

Alex Travesset, Iowa State University and Ames Laboratory

Rastko Sknepnek, Northwestern

 * integrate.npt
 * pair.morse

Carolyn Phillips, University of Michigan

 * dihedral.table
 * angle.table
 * bond.table
 * pair.dpdlj
 * pair.dpd
 * pair.dpd_conservative
 * integrate.langevin
 * bond.fene
 * pair.slj
 * Initial testing and debugging of HOOMD on Mac OS X systems

Aaron Keys, University of Michigan

  * update.enforce2d and other updates enabling to 2D simulations
  * hoomd c++ compilation helper script
  * binary restart files
  * integrate.mode_minimize_fire

Axel Kohlmeyer, David LeBard, Ben Levine, from the ICMS group at Temple University

  * pair.cgcmm
  * angle.harmonic
  * angle.cgcmm
  * dihedral.harmonic
  * improper.harmonic
  * numerous other small contributions enhancing the usability of HOOMD

Igor Morozov, Andrey Kazennov, Roman Bystryi, Joint Institute for High Temperatures of RAS (Moscow, Russia)

  * pair.eam (original implementation)

Philipp Mertmann, Ruhr University Bochum

 * charge.pppm
 * pair.ewald

Stephen Barr, Princeton University

 * charge.pppm
 * pair.ewald

Greg van Anders, Benjamin Schultz, University of Michigan

 * refactoring of ForceCompute

Eric Irrgang, University of Michigan

 * RPM packaging and daily builds

Ross Smith, University of Michigan

 * Deb packaging and daily builds

Peter Palm, Jens Glaser, Leipzig University

 * group functionality in force.constant
 * misc bug fixes
 * conversion of bond forces to template evaluator implementation

Jens Glaser, University of Michigan

 * integrate.npt anisotropic integration (mkt)
 * pair.force_shifted_lj
 * Dynamic addition/removal of bonds
 * Computation of virial and pressure tensor
 * integrate.nph
 * Framework for external potentials
 * external.periodic
 * ParticleData refactoring
 * MPI communication
 * Optimization of MPI communication for strong scaling
 * Neighborlist and pair force performance improvements (multiple threads per particle)
 * Enable cell based neighbor list on small boxes
 * Testing of angle.table and dihedral.table
 * Replicate command
 * metadata output
 * anisotropic particle integrators
 * Gay-Berne pair potential
 * pair.reaction_field
 * Rewrite of rigid body framework
 * Multi-GPU electrostatics (PPPM)
 * pair.van_der_waals
 * hpmc interaction_matrix
 * special_pair framework
 * TBB support
 * randomize integrator variables
 * GPUArray refactoring

Pavani Medapuram, University of Minnesota

 * Framework for external potentials
 * external.periodic

Brandon D. Smith, University of Michigan

 * full double precision compile time option
 * integrate.berendsen
 * pair.tersoff

Trung Dac Nguyen, University of Michigan

 * integrate.nve_rigid
 * integrate.bdnvt_rigid
 * integrate.nvt_rigid
 * integrate.npt_rigid
 * integrate.mode_minimize_rigid_fire
 * associated rigid body data structures and helper functions
 * integrate.nph_rigid

Ryan Marson, University of Michigan

 * unwrap_rigid option to dump.dcd

Kevin Silmore, Princeton University

 * OPLS dihedral

David Tarjan, University of Virginia

 * performance tweaks to the neighbor list and pair force code

Sumedh R. Risbud, James W. Swan, Massachusetts Institute of Technology

 * bug fixes for rigid body virial corrections

Michael P. Howard, Princeton University & University of Texas at Austin

 * Automatic citation list generator
 * Neighbor list memory footprint reduction
 * Bounding volume hierarchy (tree) neighbor lists
 * Stenciled cell list (stencil) neighbor lists
 * Per-type MPI ghost layer communication
 * Dynamic load balancing
 * Wall potentials extrapolated mode
 * XML dump by particle group
 * Fix references when disabling/enabling objects
 * Misc. bug fixes
 * CUDA9+V100 compatibility
 * GPU polymorphic object wrapper
 * Performance improvements to tree neighbor lists

James Antonaglia, University of Michigan

 * pair.mie

Carl Simon Adorf, University of Michigan

 * Analyzer callback
 * metadata output
 * Frenkel-Ladd bug fixes

Paul Dodd, University of Michigan

 * pair.compute_energy

Erin Teich, University of Michigan

 * addInfo callback to dump.pos

Joseph Berleant, University of Michigan

 * fix python 3.4 segfault

Matthew Spellings, University of Michigan

 * anisotropic particle integrators
 * Gay-Berne, dipole pair potentials
 * GTAR file format
 * External components in hoomd 2.x

James Proctor, University of Michigan

 * Refactor external potential framework
 * Wall potentials
 * boost python to pybind11 conversion
 * boost unit_test to upp11 conversion
 * boost signals to Nano::Signals conversion
 * Removal of misc boost library calls

Chengyu Dai, University of Michigan

 * Rewrite integrate.brownian with 3D rotational updates
 * Rewrite integrate.langevin with 3D rotational updates

Isaac Bruss, Chengyu Dai, University of Michigan

 * force.active
 * update.constraint_ellipsoid

Vyas Ramasubramani, University of Michigan

 * init.read_gsd bug fixes
 * Reverse communication for MPI
 * Enable simulation of floppy bodies that can be integrated separately but are ignored by the NeighborList
 * Enabled use of shared memory for Evaluator structs
 * Added per-type shape information to anisotropic pair potentials
 * Fix cutoff rescaling in Gay-Berne potential

Nathan Horst

 * Language and figure clarifying the dihedral angle definition.

Bryan VanSaders, University of Michigan

 * constrain.oneD
 * Constant stress mode to integrate.npt.
 * map_overlaps() in hpmc.
 * Torque options to force.constant and force.active

Ludwig Schneider, Georg-August Univeristy Goettingen

  * Constant stress flow: hoomd.md.update.mueller_plathe_flow
  * Matrix logging and hdf5 logging: hoomd.hdf5.log

Bjørnar Jensen, University of Bergen

 * Add Lennard-Jones 12-8 pair potential
 * Add Buckingham/exp-6 pair potential
 * Add special_pair Coulomb 1-4 scaling

Lin Yang, Alex Travesset, Iowa State University

  * metal.pair.eam - reworked implementation

Tim Moore, Vanderbilt University

  * angle.cosinesq
  * Documentation fixes

Bradley Dice, Avisek Das, University of Michigan

  * integrator.randomize_velocities()

Bradley Dice, Simon Adorf, University of Michigan

  * SSAGES support

Bradley Dice, University of Michigan

  * Documentation improvements
  * WSL support

Peter Schwendeman, Jens Glaser, University of Michigan

  * NVLINK optimized multi-GPU execution

Alyssa Travitz, University of Michigan

  * `get_net_force` implementation
  * bond bug fixes

Mike Henry, Boise State University

  * Documentation improvements

Pengji Zhou, University of Michigan

  * pair.fourier

Patrick Lawton, University of Michigan

  * Documentation changes

Luis Rivera-Rivera, University of Michigan

  * ``hoomd.dump.gsd.dump_shape`` implementation


Alex Yang, Vanderbilt University

  * ``hoomd.md.dihedral.harmonic`` update for phase shift

Geert Kapteijns, University of Amsterdam

  * Bug fixes.

HPMC developers
---------------

The following people contributed to the :py:mod:`hoomd.hpmc` package.

Joshua Anderson, University of Michigan - Lead developer

 * Vision
 * Initial design
 * Code review
 * NVT trial move processing (CPU / GPU)
 * Sphere shape
 * Polygon shape
 * Spheropolygon shape
 * Simple polygon shape
 * Ellipsoid shape - adaptation of Michael's Ellipsoid overlap check
 * 2D Xenocollide implementation
 * 2D GJKE implementation
 * MPI parallel domain decomposition
 * Scale distribution function pressure measurement
 * POS writer integration
 * Bounding box tree generation, query, and optimizations
 * BVH implementation of trial move processing
 * SSE and AVX intrinsics
 * `jit.patch.user` user defined patchy interactions with LLVM runtime compiled code

Eric Irrgang, University of Michigan

 * NPT updater
 * Convex polyhedron shape
 * Convex spheropolyhedron shape
 * 3D Xenocollide implementation
 * 3D GJKE implementation
 * Move size autotuner (in collaboration with Ben Schultz)
 * Densest packing compressor (in collaboration with Ben Schultz)
 * POS file utilities (in collaboration with Ben Schultz)
 * Shape union low-level implementation
 * Sphere union shape (in collaboration with Khalid Ahmed)

Ben Schultz, University of Michigan

 * Frenkel-Ladd free energy determination
 * Move size autotuner (in collaboration with Eric Irrgang)
 * Densest packing compressor (in collaboration with Eric Irrgang)
 * POS file utilities (in collaboration with Eric Irrgang)
 * Assign move size by particle type
 * Ellipsoid overlap check bug fixes

Jens Glaser, University of Michigan

 * Patchy sphere shape
 * General polyhedron shape
 * BVH implementation for countOverlaps
 * Hybrid BVH/small box trial move processing
 * Helped port the Sphinx overlap check
 * Dynamic number of particle types support
 * Implicit depletants
 * `jit.patch.user_union` user defined patchy interactions with LLVM runtime compiled code
 * Geometric Cluster Algorithm implementation
 * `convex_spheropolyhedron_union` shape class
 * `test_overlap` python API

Eric Harper, University of Michigan

 * Misc bug fixes to move size by particle type feature
 * Initial code for MPI domain decomposition

Khalid Ahmed, University of Michigan

 * Ported the Sphinx overlap check
 * Sphere union shape (in collaboration with Eric Irrgang)

Elizabeth R Chen, University of Michigan

 * Developed the Sphinx overlap check

Carl Simon Adorf, University of Michigan

 * meta data output

Samanthule Nola, University of Michigan

 * Run time determination of max_verts

Paul Dodd, Erin Teich, University of Michigan

 * External potential framework
 * Wall overlap checks
 * Lattice external potential

 Erin Teich, University of Michigan

 * Convex polyhedron union particle type

Vyas Ramasubramani, University of Michigan

 * hpmc.util.tune fixes for tuning by type
 * hpmc.update.boxmc fixes for non-orthorhombic box volume moves
 * Fixed various bugs with wall overlap checks
 * `jit.external.user` implementation
 * Refactored depletant integrators

William Zygmunt, Luis Rivera-Rivera, University of Michigan

 * Patchy interaction support in HPMC CPU integrators
 * GSD state bug fixes

DEM developers
--------------

The following people contributed to the :py:mod:`hoomd.dem` package.

Matthew Spellings, University of Michigan - Lead developer
Ryan Marson, University of Michigan

MPCD developers
---------------

The following people contributed to the :py:mod:`hoomd.mpcd` package.

Michael P. Howard, Princeton University & University of Texas at Austin - **Lead developer**

 * Design
 * Cell list and properties
 * Particle and cell communication
 * Basic streaming method
 * Slit streaming method
 * Slit pore streaming method
 * SRD and AT collision rules
 * Virtual particle filling framework
 * External force framework and block, constant, and sine forces
 * Bounce-back integrator framework

Source code
-----------

**HOOMD:** HOOMD-blue is a continuation of the HOOMD project (http://www.ameslab.gov/hoomd/). The code from the original project is used under the following license::

    Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
    Source Software License
    Copyright (c) 2008 Ames Laboratory Iowa State University
    All rights reserved.

    Redistribution and use of HOOMD, in source and binary forms, with or
    without modification, are permitted, provided that the following
    conditions are met:

    * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of the copyright holder nor the names HOOMD's
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

    Disclaimer

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
    CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
    AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    THE POSSIBILITY OF SUCH DAMAGE.

**Sockets code** from VMD is used for the IMDInterface to VMD (http://www.ks.uiuc.edu/Research/vmd/) - Used under the UIUC Open Source License.

**Molfile plugin code** from VMD is used for generic file format reading and writing - Used under the UIUC Open Source License::

    University of Illinois Open Source License
    Copyright 2006 Theoretical and Computational Biophysics Group,
    All rights reserved.

    Developed by: Theoretical and Computational Biophysics Group
                  University of Illinois at Urbana-Champaign
                  http://www.ks.uiuc.edu/

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the Software), to deal with
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to
    do so, subject to the following conditions:

    Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimers.

    Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimers in the documentation
    and/or other materials provided with the distribution.

    Neither the names of Theoretical and Computational Biophysics Group,
    University of Illinois at Urbana-Champaign, nor the names of its contributors
    may be used to endorse or promote products derived from this Software without
    specific prior written permission.

    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
    THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS WITH THE SOFTWARE.


**XML parsing** is performed with XML.c from http://www.applied-mathematics.net/tools/xmlParser.html - Used under the BSD License::

    Copyright (c) 2002, Frank Vanden Berghen<br>
    All rights reserved.<br>
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

     - Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
     - Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
     - Neither the name of the Frank Vanden Berghen nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**Saru** is used for random number generation - Used under the following license::

    Copyright (c) 2008 Steve Worley < m a t h g e e k@(my last name).com >

    Permission to use, copy, modify, and distribute this software for any
    purpose with or without fee is hereby granted, provided that the above
    copyright notice and this permission notice appear in all copies.

    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Some **CUDA API headers** are included in the HOOMD-blue source code for code compatibility in CPU only builds - Used under the following license::

    Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.

    NOTICE TO USER:

    This source code is subject to NVIDIA ownership rights under U.S. and
    international Copyright laws.  Users and possessors of this source code
    are hereby granted a nonexclusive, royalty-free license to use this code
    in individual and commercial software.

    NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
    CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
    IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
    REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
    IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
    OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
    OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
    OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
    OR PERFORMANCE OF THIS SOURCE CODE.

    U.S. Government End Users.   This source code is a "commercial item" as
    that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
    "commercial computer  software"  and "commercial computer software
    documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
    and is provided to the U.S. Government only as a commercial end item.
    Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
    227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
    source code with only those rights set forth herein.

    Any use of this source code in individual and commercial software must
    include, in the user documentation and internal comments to the code,
    the above Disclaimer and U.S. Government End Users Notice.

FFTs on the CPU reference implementation of PPPM are performed using **kissFFT** from http://sourceforge.net/projects/kissfft/,
used under the following license::

    Copyright (c) 2003-2010 Mark Borgerding

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice, this
    list of conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    * Neither the author nor the names of any contributors may be used to endorse or
    promote products derived from this software without specific prior written
    permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

ModernGPU source code is embedded in HOOMD's package and is used for various tasks: http://nvlabs.github.io/moderngpu/::

    Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the NVIDIA CORPORATION nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CUB 1.4.1 source code is embedded in HOOMD's package and is used for various tasks: http://nvlabs.github.io/cub/::

    Copyright (c) 2011, Duane Merrill.  All rights reserved.
    Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the NVIDIA CORPORATION nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Eigen 3.2.5 (http://eigen.tuxfamily.org/) is embedded in HOOMD's package and is made available under the
Mozilla Public License v.2.0 (http://mozilla.org/MPL/2.0/). Its linear algebra routines are used for dynamic load balancing. Source code is available through the [downloads](http://glotzerlab.engin.umich.edu/hoomd-blue/download.html).

A constrained least-squares problem is solved for dynamic load balancing using **BVLSSolver**, which is embedded
in HOOMD's package and is made available under the following license::

    Copyright (c) 2015, Michael P. Howard. All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        1. Redistributions of source code must retain the above copyright
           notice, this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
           notice, this list of conditions and the following disclaimer in the
           documentation and/or other materials provided with the distribution.

        3. Neither the name of the copyright holder nor the names of its
           contributors may be used to endorse or promote products derived from
           this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
    OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
    OF SUCH DAMAGE.

libgetar is used to read and write GTAR files. Used under the MIT license::

    Copyright (c) 2014-2016 The Regents of the University of Michigan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

pybind11 is used to provide python bindings for C++ classes. Used under the BSD license::

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    You are under no obligation whatsoever to provide any bug fixes, patches, or
    upgrades to the features, functionality or performance of the source code
    ("Enhancements") to anyone; however, if you choose to make your Enhancements
    available either publicly, or directly to the author of this software, without
    imposing a separate written license agreement for such Enhancements, then you
    hereby grant the following license: a non-exclusive, royalty-free perpetual
    license to install, use, modify, prepare derivative works, incorporate into
    other computer software, distribute, and sublicense such enhancements or
    derivative works thereof, in binary and source code form.

cereal is used to serialize C++ objects for transmission over MPI. Used under the BSD license::

    Copyright (c) 2014, Randolph Voorhies, Shane Grant
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of cereal nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL RANDOLPH VOORHIES OR SHANE GRANT BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Random123 is used to generate random numbers and is used under the following license::

    Copyright 2010-2012, D. E. Shaw Research.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions, and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions, and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of D. E. Shaw Research nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

A CUDA [neighbor](https://github.com/mphoward/neighbor) search library is
used under the Modified BSD license::

    Copyright (c) 2018-2019, Michael P. Howard. All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Libraries
---------

HOOMD-blue links to the following libraries:

 * python - Used under the Python license (http://www.python.org/psf/license/)
 * cuFFT - Used under the NVIDIA CUDA toolkit license (http://docs.nvidia.com/cuda/eula/index.html)
