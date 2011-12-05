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

#include <iostream>

#include <boost/shared_ptr.hpp>

#include "IntegratorTwoStep.h"
#include "TwoStepBDNVT.h"
#include "TwoStepBerendsen.h"
#ifdef ENABLE_CUDA
#include "TwoStepBerendsenGPU.h"
#endif

#include "Initializers.h"

#include <math.h>

using namespace std;
using namespace boost;

/*! \file test_berendsen_updater.cc
    \brief Implements unit tests for TwoStepBerendsen and descendants
    \ingroup unit_tests
*/

//! name the boost unit test module
#define BOOST_TEST_MODULE Berendsen_UpdaterTests
#include "boost_utf_configure.h"

//! Apply the thermostat to 1000 particles in an ideal gas
template <class Berendsen>
void berend_updater_lj_tests(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    // check that the berendsen thermostat applied to a system of 1000 LJ particles produces the correct average temperature
    // Build a 1000 particle system with particles scattered on the x, y, and z axes.
    RandomInitializer rand_init(1000, Scalar(0.05), Scalar(1.3), "A");
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(rand_init, exec_conf));

    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    Scalar deltaT = Scalar(0.002);
    Scalar Temp = Scalar(2.0);

    shared_ptr<ComputeThermo> thermo(new ComputeThermo(sysdef, group_all));
    thermo->setNDOF(3*1000-3);
    shared_ptr<VariantConst> T_variant(new VariantConst(Temp));
    shared_ptr<VariantConst> T_variant2(new VariantConst(1.0));

    shared_ptr<TwoStepBerendsen> two_step_berendsen(new Berendsen(sysdef, group_all, thermo, 1.0, T_variant));
    shared_ptr<IntegratorTwoStep> berendsen_up(new IntegratorTwoStep(sysdef, deltaT));
    berendsen_up->addIntegrationMethod(two_step_berendsen);

    shared_ptr<TwoStepBDNVT> two_step_bdnvt(new TwoStepBDNVT(sysdef, group_all, T_variant2, 268, 1));
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);
    bdnvt_up->prepRun(0);

    int i;

    // ramp up to temp
    for (i = 0; i < 10000; i++)
        bdnvt_up->update(i);

    Scalar AvgT = Scalar(0);

    // equilibrate with berend
    berendsen_up->prepRun(0);

    for (i = 0; i < 5000; i++)
        berendsen_up->update(i);

    for (; i < 10000; i++)
        {
        berendsen_up->update(i);

        if (i % 10 == 0)
            {
            thermo->compute(i);
            AvgT += thermo->getTemperature();
            }
        }
    AvgT /= Scalar(5000.0/10.0);

    MY_BOOST_CHECK_CLOSE(AvgT, 2.0, 1);

    // Resetting the Temperature to 1.0
    two_step_berendsen->setT(T_variant2);

    AvgT = Scalar(0);
    for (i = 0; i < 5000; i++)
        berendsen_up->update(i);

    for (; i < 10000; i++)
        {
        if (i % 10 == 0)
            AvgT += thermo->getTemperature();

        berendsen_up->update(i);
        }
    AvgT /= Scalar(5000.0/10.0);

    MY_BOOST_CHECK_CLOSE(AvgT, 1.0, 1);
    }

//! extended LJ-liquid test for the base class
BOOST_AUTO_TEST_CASE( TwoStepBerendsen_LJ_tests )
    {
    berend_updater_lj_tests<TwoStepBerendsen>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::CPU)));
    }

#ifdef ENABLE_CUDA
//! extended LJ-liquid test for the GPU class
BOOST_AUTO_TEST_CASE( TwoStepBerendsenGPU_LJ_tests )
    {
    berend_updater_lj_tests<TwoStepBerendsenGPU>(boost::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)));
    }
#endif


#ifdef WIN32
#pragma warning( pop )
#endif

