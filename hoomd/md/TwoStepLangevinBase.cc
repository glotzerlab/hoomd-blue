/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

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



#include "TwoStepLangevinBase.h"

#include <boost/python.hpp>
using namespace boost::python;
using namespace std;

/*! \file TwoStepLangevinBase.h
    \brief Contains code for the TwoStepLangevinBase class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
*/
TwoStepLangevinBase::TwoStepLangevinBase(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda)
    : IntegrationMethodTwoStep(sysdef, group), m_T(T), m_seed(seed), m_use_lambda(use_lambda), m_lambda(lambda)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepLangevinBase" << endl;

    if (use_lambda)
        m_exec_conf->msg->notice(2) << "integrate.langevin/bd is determining gamma from particle diameters" << endl;
    else
        m_exec_conf->msg->notice(2) << "integrate.langevin/bd is using specified gamma values" << endl;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GPUVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
    m_gamma.swap(gamma);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.size(); i++)
        h_gamma.data[i] = Scalar(1.0);
        
    // allocate memory for the per-type gamma_r storage and initialize them to 0.0 (no rotational noise by default)
    GPUVector<Scalar> gamma_r(m_pdata->getNTypes(), m_exec_conf);
    m_gamma_r.swap(gamma_r);
    ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma_r.size(); i++)
        h_gamma_r.data[i] = Scalar(1.0);

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_num_type_change_connection = m_pdata->connectNumTypesChange(boost::bind(&TwoStepLangevinBase::slotNumTypesChange, this));
    }

TwoStepLangevinBase::~TwoStepLangevinBase()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepLangevinBase" << endl;
    m_num_type_change_connection.disconnect();
    }

void TwoStepLangevinBase::slotNumTypesChange()
    {
    // skip the reallocation if the number of types does not change
    // this keeps old parameters when restoring a snapshot
    // it will result in invalid coefficients if the snapshot has a different type id -> name mapping
    if (m_pdata->getNTypes() == m_gamma.size())
        return;

    // re-allocate memory for the per-type gamma storage and initialize them to 1.0
    unsigned int old_ntypes = m_gamma.size();
    m_gamma.resize(m_pdata->getNTypes());
    m_gamma_r.resize(m_pdata->getNTypes());
    
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);
    
    for (unsigned int i = old_ntypes; i < m_gamma.size(); i++)
        {
        h_gamma.data[i] = Scalar(1.0);
        h_gamma_r.data[i] = Scalar(1.0);
        }
    }

/*! \param typ Particle type to set gamma for
    \param gamma The gamma value to set
*/
void TwoStepLangevinBase::setGamma(unsigned int typ, Scalar gamma)
    {
    // check for user errors
    if (m_use_lambda)
        {
        m_exec_conf->msg->error() << "Trying to set gamma when it is set to be the diameter! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Trying to set gamma for a non existent type! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }
    
    
/*! \param typ Particle type to set gamma_r (2D rotational noise) for
    \param gamma The gamma_r value to set
*/    
void TwoStepLangevinBase::setGamma_r(unsigned int typ, Scalar gamma_r)
    {
    // check for user errors
    if (gamma_r < 0)
        {
        m_exec_conf->msg->error() << "gamma_r should be positive or 0! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Trying to set gamma_r for a non existent type! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }

    ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);
    h_gamma_r.data[typ] = gamma_r;
    }

void export_TwoStepLangevinBase()
    {
    class_<TwoStepLangevinBase, boost::shared_ptr<TwoStepLangevinBase>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepLangevinBase", init< boost::shared_ptr<SystemDefinition>,
                                boost::shared_ptr<ParticleGroup>,
                                boost::shared_ptr<Variant>,
                                unsigned int,
                                bool,
                                Scalar
                                >())
        .def("setT", &TwoStepLangevinBase::setT)
        .def("setGamma", &TwoStepLangevinBase::setGamma)
        .def("setGamma_r", &TwoStepLangevinBase::setGamma_r)
        ;
    }
