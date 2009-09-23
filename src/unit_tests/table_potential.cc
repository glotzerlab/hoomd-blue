/*!
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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

//! Name the unit test module
#define BOOST_TEST_MODULE TablePotentialTests
#include "boost_utf_configure.h"

#include <fstream>

#include "TablePotential.h"
#include "BinnedNeighborList.h"

using namespace std;
using namespace boost;

/*! \file table_potential_test.cc
    \brief Implements unit tests for TablePotential and descendants
    \ingroup unit_tests
*/

BOOST_AUTO_TEST_CASE(potential_writer)
	{
	#ifdef CUDA
	g_gpu_error_checking = true;
	#endif
	
	// this 2-particle test is just to get a plot of the potential and force vs r cut
	shared_ptr<SystemDefinition> sysdef_2(new SystemDefinition(2, BoxDim(1000.0), 1, 0, 0, 0, 0, ExecutionConfiguration()));
    shared_ptr<ParticleData> pdata_2 = sysdef_2->getParticleData();
    
	ParticleDataArrays arrays = pdata_2->acquireReadWrite();
	arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
	arrays.x[1] = Scalar(0.9); arrays.y[1] = arrays.z[1] = 0.0;
	pdata_2->release();
	shared_ptr<NeighborList> nlist_2(new NeighborList(sysdef_2, Scalar(7.0), Scalar(0.8)));
	shared_ptr<TablePotential> fc(new TablePotential(sysdef_2, nlist_2, 1000));

    // provide a basic potential and "force"
    vector<Scalar> V, F;
    // 5 point test
//    V.push_back(10.0);  F.push_back(-10.0/1.0);
//    V.push_back(15.0);  F.push_back(-15.0/2.0);
//    V.push_back(5.0);   F.push_back(-5.0/3.0);
//    V.push_back(8.0);   F.push_back(-8.0/4.0);
//    V.push_back(18.0);  F.push_back(-18.0/5.0);

    // 1000 point lj test
    Scalar delta_r = (5.0 - 0.5) / (999);
    for (unsigned int i = 0; i < 1000; i++)
        {
        Scalar r = 0.5 + delta_r * Scalar(i);
        V.push_back(4.0 * (pow(1.0 / r, 12) - pow(1.0 / r, 6)));
        F.push_back(4.0 * (12.0 * pow(1.0 / r, 14) - 6 * pow(1.0 / r, 8)));
        }
    
	fc->setTable(0, 0, V, F, 0.5, 5.0);
	
	ofstream f("table_dat.m");
	f << "table = [";
	unsigned int count = 0;	
	for (float r = 0.95; r <= 5.0; r+= 0.001)
		{
		// set the distance
		ParticleDataArrays arrays = pdata_2->acquireReadWrite();
		arrays.x[0] = arrays.y[0] = arrays.z[0] = 0.0;
		arrays.x[1] = Scalar(r); arrays.y[1] = arrays.z[1] = 0.0;
		pdata_2->release();
		
		// compute the forces
		fc->compute(count);
		count++;
	
		ForceDataArrays force_arrays = fc->acquire();
		f << r << " " << force_arrays.fx[0] << " " << fc->calcEnergySum() << " ; " << endl;	
		}
	f << "];" << endl;
	f.close();
	}