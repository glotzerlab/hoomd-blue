/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

/*! \file SnapshotSystemData.h
    \brief Defines the SnapshotSystemData class 
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __SNAPSHOT_SYSTEM_DATA_H__
#define __SNAPSHOT_SYSTEM_DATA_H__

#include "BoxDim.h"
#include "ParticleData.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "IntegratorData.h"

/*! \ingroup data_structs
*/

//! Structure for initializing system data
/*! A snapshot is used for multiple purposes:
 * 1. for initializing the system
 * 2. during the simulation, e.g. to dump the system state or to analyze it
 *
 * Snapshots are temporary data-structures, they are only used for passing around data.
 *
 * A SnapshotSystemData is just a super-structure that holds snapshots of other data, such
 * as particles, bonds, rigid bodies, etc. It is used by the SystemDefinition class to initially
 * set up these data structures, and can also be obtained from an object of that class to
 * analyze the current system state.
 *
 * \ingroup data_structs
 */
struct SnapshotSystemData {
    unsigned int dimensions;               //!< The dimensionality of the system
    BoxDim global_box;                     //!< The dimensions of the simulation box
    SnapshotParticleData particle_data;    //!< The particle data
    SnapshotBondData bond_data;            //!< The bond data
    SnapshotAngleData angle_data;          //!< The angle data
    SnapshotDihedralData dihedral_data;    //!< The dihedral data
    SnapshotDihedralData improper_data;    //!< The improper data
    std::vector<IntegratorVariables> integrator_data;  //!< The integrator data
   
    //! Constructor
    SnapshotSystemData()
        {
        dimensions = 3;
        }
    };

//! Export SnapshotParticleData to python
void export_SnapshotSystemData();

#endif

