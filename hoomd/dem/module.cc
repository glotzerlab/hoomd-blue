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

// Include the defined classes that are to be exported to python
#include <hoomd/HOOMDMath.h>

#include "NoFriction.h"

#include "DEMEvaluator.h"
#include "WCAPotential.h"
#include "SWCAPotential.h"
#include "DEM2DForceCompute.h"
#include "DEM2DForceComputeGPU.h"
#include "DEM3DForceCompute.h"
#include "DEM3DForceComputeGPU.h"

#include <iterator>

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

void export_params();

void export_NF_WCA_2D();
void export_NF_WCA_3D();
void export_NF_SWCA_3D();
void export_NF_SWCA_2D();

BOOST_PYTHON_MODULE(_dem)
{
    export_params();
    export_NF_WCA_2D();
    export_NF_WCA_3D();
    export_NF_SWCA_2D();
    export_NF_SWCA_3D();
}

// Export all of the parameter wrapper objects to the python interface
void export_params()
{
    class_<NoFriction<Scalar> >("NoFriction");

    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar> > WCA;
    typedef SWCAPotential<Scalar, Scalar4, NoFriction<Scalar> > SWCA;

    class_<WCA>("WCAPotential", init<Scalar, NoFriction<Scalar> >());
    class_<SWCA>("SWCAPotential", init<Scalar, NoFriction<Scalar> >());
}
