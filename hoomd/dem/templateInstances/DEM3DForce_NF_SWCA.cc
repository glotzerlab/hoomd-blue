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

#include <hoomd/HOOMDMath.h>

#include "../DEMEvaluator.h"
#include "../NoFriction.h"
#include "../WCAPotential.h"
#include "../SWCAPotential.h"
#include "../DEM2DForceCompute.h"
#include "../DEM2DForceComputeGPU.h"
#include "../DEM3DForceCompute.h"
#include "../DEM3DForceComputeGPU.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

void export_NF_SWCA_3D()
{
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > SWCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, SWCA> SWCA_DEM_3D;

    class_<SWCA_DEM_3D, boost::shared_ptr<SWCA_DEM_3D>, bases<ForceCompute>, boost::noncopyable >
        ("SWCADEM3D", init< boost::shared_ptr<SystemDefinition>,
         boost::shared_ptr<NeighborList>, Scalar, SWCA>())
        .def("setParams", &SWCA_DEM_3D::setParams)
        .def("setRcut", &SWCA_DEM_3D::setRcut)
        ;

#ifdef ENABLE_CUDA
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, SWCA> SWCA_DEM_3D_GPU;

    class_<SWCA_DEM_3D_GPU, boost::shared_ptr<SWCA_DEM_3D_GPU>,
           bases<SWCA_DEM_3D>, boost::noncopyable >
        ("SWCADEM3DGPU", init< boost::shared_ptr<SystemDefinition>,
         boost::shared_ptr<NeighborList>, Scalar, SWCA>())
        .def("setParams", &SWCA_DEM_3D_GPU::setParams)
        .def("setRcut", &SWCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &SWCA_DEM_3D_GPU::setAutotunerParams)
        ;
#endif
}
