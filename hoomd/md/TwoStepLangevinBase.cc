// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepLangevinBase.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;

/** @param sysdef SystemDefinition this method will act on. Must not be NULL.
    @param group The group of particles this integration method is to work on
    @param T Temperature set point as a function of time
    @param seed Random seed to use in generating random numbers
    @note All ranks other than 0 ignore the seed input and use the value of rank 0.
*/
TwoStepLangevinBase::TwoStepLangevinBase(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           unsigned int seed)
    : IntegrationMethodTwoStep(sysdef, group), m_T(T), m_seed(seed), m_use_alpha(false),
      m_alpha(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepLangevinBase" << endl;

    // In case of MPI run, every rank should be initialized with the same seed.
    // For simplicity we broadcast the seed of rank 0 to all ranks.

    #ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
    #endif

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GlobalVector<Scalar> gamma(m_pdata->getNTypes(), m_exec_conf);
    m_gamma.swap(gamma);
    TAG_ALLOCATION(m_gamma);

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.size(); i++)
        h_gamma.data[i] = Scalar(1.0);

    // allocate memory for the per-type gamma_r storage
    GlobalVector<Scalar3> gamma_r(m_pdata->getNTypes(), m_exec_conf);
    m_gamma_r.swap(gamma_r);
    TAG_ALLOCATION(m_gamma_r);

    #if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
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

void TwoStepLangevinBase::setGamma(const std::string& type_name, Scalar gamma)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }

Scalar TwoStepLangevinBase::getGamma(const std::string& type_name)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    return h_gamma.data[typ];
    }

void TwoStepLangevinBase::setGammaR(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw invalid_argument("gamma_r values must be 3-tuples");
        }

    Scalar3 gamma_r;
    gamma_r.x = pybind11::cast<Scalar>(v[0]);
    gamma_r.y = pybind11::cast<Scalar>(v[1]);
    gamma_r.z = pybind11::cast<Scalar>(v[2]);

    // check for user errors
    if (gamma_r.x < 0 || gamma_r.y < 0 || gamma_r. z < 0)
        {
        throw invalid_argument("gamma_r elements must be >= 0");
        }
    if (typ >= m_pdata->getNTypes())
        {
        throw invalid_argument("Type does not exist");
        }

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);
    h_gamma_r.data[typ] = gamma_r;
    }

pybind11::tuple TwoStepLangevinBase::getGammaR(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::readwrite);
    Scalar3 gamma_r = h_gamma_r.data[typ];
    v.append(gamma_r.x);
    v.append(gamma_r.y);
    v.append(gamma_r.z);
    return pybind11::tuple(v);
    }

void TwoStepLangevinBase::setAlpha(pybind11::object alpha)
    {
    if (alpha.is_none())
        {
        m_use_alpha = false;
        }
    else
        {
        m_use_alpha = true;
        m_alpha = pybind11::cast<Scalar>(alpha);
        }
    }

pybind11::object TwoStepLangevinBase::getAlpha()
    {
    pybind11::object result;
    if (m_use_alpha)
        {
        result = pybind11::cast(m_alpha);
        }
    else
        {
        result = pybind11::none();
        }
    return result;
    }

void export_TwoStepLangevinBase(py::module& m)
    {
    py::class_<TwoStepLangevinBase, IntegrationMethodTwoStep, std::shared_ptr<TwoStepLangevinBase> >(m, "TwoStepLangevinBase")
        .def(py::init< std::shared_ptr<SystemDefinition>,
                                std::shared_ptr<ParticleGroup>,
                                std::shared_ptr<Variant>,
                                unsigned int
                                >())
        .def_property("kT", &TwoStepLangevinBase::getT, &TwoStepLangevinBase::setT)
        .def("setGamma", &TwoStepLangevinBase::setGamma)
        .def("getGamma", &TwoStepLangevinBase::getGamma)
        .def("setGammaR", &TwoStepLangevinBase::setGammaR)
        .def("getGammaR", &TwoStepLangevinBase::getGammaR)
        .def_property("alpha", &TwoStepLangevinBase::getAlpha,
                      &TwoStepLangevinBase::setAlpha)
        .def_property_readonly("seed", &TwoStepLangevinBase::getSeed)
        ;
    }
