// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander



#include "TwoStepLangevinBase.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
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
    \note All ranks other than 0 ignore the seed input and use the value of rank 0.
*/
TwoStepLangevinBase::TwoStepLangevinBase(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
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

    // In case of MPI run, every rank should be initialized with the same seed.
    // For simplicity we broadcast the seed of rank 0 to all ranks.

    #ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
    #endif

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GlobalVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
    m_gamma.swap(gamma);
    TAG_ALLOCATION(m_gamma);

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.size(); i++)
        h_gamma.data[i] = Scalar(1.0);

    // allocate memory for the per-type gamma_r storage and initialize them to 0.0 (no rotational noise by default)
    GlobalVector<Scalar3> gamma_r(m_pdata->getNTypes(), m_exec_conf);
    m_gamma_r.swap(gamma_r);
    TAG_ALLOCATION(m_gamma_r);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_gamma.get(), sizeof(Scalar)*m_gamma.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_gamma_r.get(), sizeof(Scalar3)*m_gamma_r.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        }
    #endif

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma_r.size(); i++)
        h_gamma_r.data[i] = make_scalar3(1.0,1.0,1.0);

    // connect to the ParticleData to receive notifications when the maximum number of particles changes
    m_pdata->getNumTypesChangeSignal().connect<TwoStepLangevinBase, &TwoStepLangevinBase::slotNumTypesChange>(this);
    }

TwoStepLangevinBase::~TwoStepLangevinBase()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepLangevinBase" << endl;
    m_pdata->getNumTypesChangeSignal().disconnect<TwoStepLangevinBase, &TwoStepLangevinBase::slotNumTypesChange>(this);
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
    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);

    for (unsigned int i = old_ntypes; i < m_gamma.size(); i++)
        {
        h_gamma.data[i] = Scalar(1.0);
        h_gamma_r.data[i] = make_scalar3(1.0,1.0,1.0);
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
void TwoStepLangevinBase::setGamma_r(unsigned int typ, Scalar3 gamma_r)
    {
    // check for user errors
    if (gamma_r.x < 0 || gamma_r.y < 0 || gamma_r. z < 0)
        {
        m_exec_conf->msg->error() << "gamma_r.(x,y,z) should be positive or 0! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "Trying to set gamma_r for a non existent type! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepLangevinBase");
        }

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);
    h_gamma_r.data[typ] = gamma_r;
    }

void export_TwoStepLangevinBase(py::module& m)
    {
    py::class_<TwoStepLangevinBase, std::shared_ptr<TwoStepLangevinBase> >(m, "TwoStepLangevinBase", py::base<IntegrationMethodTwoStep>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                                std::shared_ptr<ParticleGroup>,
                                std::shared_ptr<Variant>,
                                unsigned int,
                                bool,
                                Scalar
                                >())
        .def("setT", &TwoStepLangevinBase::setT)
        .def("setGamma", &TwoStepLangevinBase::setGamma)
        .def("setGamma_r", &TwoStepLangevinBase::setGamma_r)
        ;
    }
