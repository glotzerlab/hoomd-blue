/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifndef _HOOMD_H
#define _HOOMD_H

#include "hoomd_config.h"

// math setup
#include "HOOMDMath.h"

// data structures
#include "GPUArray.h"
#include "ParticleData.h"
#include "AngleData.h"
#include "BondData.h"
#include "DihedralData.h"
#include "IntegratorData.h"
#include "ParticleGroup.h"
#include "WallData.h"

#include "ExecutionConfiguration.h"
#include "SystemDefinition.h"

// initializers
#include "Initializers.h"
#include "RandomGenerator.h"
#include "HOOMDBinaryInitializer.h"
#include "HOOMDInitializer.h"

// base classes
#include "Analyzer.h"
#include "Compute.h"
#include "Updater.h"

// analyzers
#include "DCDDumpWriter.h"
#include "HOOMDBinaryDumpWriter.h"
#include "HOOMDDumpWriter.h"
#include "IMDInterface.h"
#include "Logger.h"
#include "MOL2DumpWriter.h"
#include "MSDAnalyzer.h"
#include "PDBDumpWriter.h"

// computes
#include "ForceCompute.h"
#include "CellList.h"
#include "NeighborList.h"
#include "NeighborListBinned.h"
#include "ConstForceCompute.h"

#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#include "NeighborListGPUBinned.h"
#endif

// pair potentials
#include "TablePotential.h"
#include "CGCMMForceCompute.h"
#include "AllPairPotentials.h"

#ifdef ENABLE_CUDA
#include "CGCMMForceComputeGPU.h"
#include "TablePotentialGPU.h"
#endif

// bond potentials
#include "FENEBondForceCompute.h"
#include "HarmonicBondForceCompute.h"

#ifdef ENABLE_CUDA
#include "HarmonicBondForceComputeGPU.h"
#include "FENEBondForceComputeGPU.h"
#endif

// angle potentials
#include "CGCMMAngleForceCompute.h"
#include "HarmonicAngleForceCompute.h"

#ifdef ENABLE_CUDA
#include "CGCMMAngleForceComputeGPU.h"
#include "HarmonicAngleForceComputeGPU.h"
#endif

// dihedral/improper potentials
#include "HarmonicDihedralForceCompute.h"
#include "HarmonicImproperForceCompute.h"

#ifdef ENABLE_CUDA
#include "HarmonicDihedralForceComputeGPU.h"
#include "HarmonicImproperForceComputeGPU.h"
#endif

// wall potentials
#include "LJWallForceCompute.h"

// system
#include "System.h"

// updaters
#include "BoxResizeUpdater.h"
#include "Enforce2DUpdater.h"
#include "SFCPackUpdater.h"
#include "TempRescaleUpdater.h"
#include "ZeroMomentumUpdater.h"
#include "FIREEnergyMinimizer.h"
#ifdef ENABLE_CUDA
#include "Enforce2DUpdaterGPU.h"
#include "FIREEnergyMinimizerGPU.h"
#endif

// integrators
#include "IntegrationMethodTwoStep.h"
#include "Integrator.h"
#include "IntegratorTwoStep.h"
#include "TwoStepBDNVT.h"
#include "TwoStepNPT.h"
#include "TwoStepNVE.h"
#include "TwoStepNVT.h"
#include "ClockSource.h"

#ifdef ENABLE_CUDA
#include "TwoStepBDNVTGPU.h"
#include "TwoStepNPTGPU.h"
#include "TwoStepNVEGPU.h"
#include "TwoStepNVTGPU.h"
#endif

// utility classes
#include "FileFormatManager.h"
#include "FileFormatProxy.h"
#include "Index1D.h"
#include "MolFilePlugin.h"
#include "Profiler.h"
#include "SignalHandler.h"
#include "Variant.h"
#include "HOOMDVersion.h"

#endif

