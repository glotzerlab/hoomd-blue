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

// Maintainer: joaander

#ifndef _HOOMD_H
#define _HOOMD_H

#include "hoomd_config.h"

#include "HOOMDMath.h"
#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "RigidData.h"
#include "SystemDefinition.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"
#include "ExecutionConfiguration.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "HOOMDBinaryInitializer.h"
#include "RandomGenerator.h"
#include "Compute.h"
#include "CellList.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "ConstForceCompute.h"
#include "ConstExternalFieldDipoleForceCompute.h"
#include "HarmonicAngleForceCompute.h"
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"
#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"
#include "TablePotential.h"
#include "LJWallForceCompute.h"
#include "AllPairPotentials.h"
#include "AllBondPotentials.h"
#include "ComputeThermo.h"
#include "ComputeThermoGPU.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "HOOMDBinaryDumpWriter.h"
#include "PDBDumpWriter.h"
#include "MOL2DumpWriter.h"
#include "DCDDumpWriter.h"
#include "Logger.h"
#include "MSDAnalyzer.h"
#include "Updater.h"
#include "Integrator.h"
#include "IntegratorTwoStep.h"
#include "IntegrationMethodTwoStep.h"
#include "TwoStepNVE.h"
#include "TwoStepNVT.h"
#include "TwoStepBDNVT.h"
#include "TwoStepNPT.h"
#include "TwoStepNPH.h"
#include "TwoStepBerendsen.h"
#include "TwoStepNVERigid.h" 
#include "TwoStepNVTRigid.h"
#include "TwoStepNPTRigid.h"  
#include "TwoStepBDNVTRigid.h" 
#include "TempRescaleUpdater.h"
#include "ZeroMomentumUpdater.h"
#include "FIREEnergyMinimizer.h"
#include "FIREEnergyMinimizerRigid.h"
#include "SFCPackUpdater.h"
#include "BoxResizeUpdater.h"
#include "Enforce2DUpdater.h"
#include "System.h"
#include "Variant.h"
#include "EAMForceCompute.h"
#include "ConstraintSphere.h"
#include "PotentialPairDPDThermo.h"
#include "PotentialPair.h"
#include "PPPMForceCompute.h"
#include "AllExternalPotentials.h"


// include GPU classes
#ifdef ENABLE_CUDA
#include <cuda.h>
#include "CellListGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTGPU.h"
#include "TwoStepBDNVTGPU.h"
#include "TwoStepNPTGPU.h"
#include "TwoStepNPHGPU.h"
#include "TwoStepBerendsenGPU.h"
#include "TwoStepNVERigidGPU.h" 
#include "TwoStepNVTRigidGPU.h" 
#include "TwoStepNPTRigidGPU.h" 
#include "TwoStepBDNVTRigidGPU.h" 
#include "NeighborListGPU.h"
#include "NeighborListGPUBinned.h"
#include "CGCMMForceComputeGPU.h"
//#include "ConstExternalFieldDipoleForceComputeGPU.h"
#include "TablePotentialGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#include "HarmonicDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#include "CGCMMAngleForceComputeGPU.h"
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerRigidGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#include "EAMForceComputeGPU.h"
#include "ConstraintSphereGPU.h"
#include "PotentialPairGPU.h"
#include "PPPMForceComputeGPU.h"
#endif

#include "SignalHandler.h"

#include "HOOMDVersion.h"
#include "PathUtils.h"

#endif

