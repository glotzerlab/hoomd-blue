// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GlobalArray.h"
#include <memory>
#include <stdexcept>

/*! \file PotentialExternal.h
    \brief Declares a class for computing an external force field
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __POTENTIAL_EXTERNAL_H__
#define __POTENTIAL_EXTERNAL_H__

namespace hoomd
    {
namespace md
    {
//! Applys an external force to particles based on position
/*! \ingroup computes
 */
template<class evaluator> class PotentialExternal : public ForceCompute
    {
    public:
    //! Constructs the compute
    PotentialExternal<evaluator>(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~PotentialExternal<evaluator>();

    //! type of external potential parameters
    typedef typename evaluator::param_type param_type;
    typedef typename evaluator::field_type field_type;

    //! Sets parameters of the evaluator
    pybind11::object getParams(std::string type);

    //! set the potential parameters via cpp arguments
    void setParams(unsigned int type, const param_type& params);

    //! set the potential parameters via python arguments
    void setParamsPython(std::string typ, pybind11::object params);

    //! make sure the type index is within range
    void validateType(unsigned int type, std::string action);

    //! set the field type of the evaluator
    void setField(field_type field);

    protected:
    GPUArray<param_type> m_params; //!< Array of per-type parameters
    GPUArray<field_type> m_field;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! Constructor
    \param sysdef system definition
*/
template<class evaluator>
PotentialExternal<evaluator>::PotentialExternal(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
    {
    GPUArray<param_type> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);

    GPUArray<field_type> field(1, m_exec_conf);
    m_field.swap(field);
    }

/*! Destructor
 */
template<class evaluator> PotentialExternal<evaluator>::~PotentialExternal() { }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator> void PotentialExternal<evaluator>::computeForces(uint64_t timestep)
    {
    if (m_prof)
        m_prof->push("PotentialExternal");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    ArrayHandle<field_type> h_field(m_field, access_location::host, access_mode::read);
    const field_type& field = *(h_field.data);

    const BoxDim& box = m_pdata->getGlobalBox();
    PDataFlags flags = this->m_pdata->getFlags();

    if (flags[pdata_flag::external_field_virial])
        {
        bool virial_terms_defined = evaluator::requestFieldVirialTerm();
        if (!virial_terms_defined)
            {
            this->m_exec_conf->msg->error()
                << "The required virial terms are not defined for the current setup." << std::endl;
            throw std::runtime_error("NPT is not supported for requested features");
            }
        }

    unsigned int nparticles = m_pdata->getN();

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    // for each of the particles
    for (unsigned int idx = 0; idx < nparticles; idx++)
        {
        // get the current particle properties
        Scalar3 X = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        Scalar3 F;
        Scalar energy;
        Scalar virial[6];

        param_type params = h_params.data[type];
        evaluator eval(X, box, params, field);

        if (evaluator::needsDiameter())
            {
            Scalar di = h_diameter.data[idx];
            eval.setDiameter(di);
            }
        if (evaluator::needsCharge())
            {
            Scalar qi = h_charge.data[idx];
            eval.setCharge(qi);
            }
        eval.evalForceEnergyAndVirial(F, energy, virial);

        // apply the constraint force
        h_force.data[idx].x = F.x;
        h_force.data[idx].y = F.y;
        h_force.data[idx].z = F.z;
        h_force.data[idx].w = energy;
        for (int k = 0; k < 6; k++)
            h_virial.data[k * m_virial_pitch + idx] = virial[k];
        }

    if (m_prof)
        m_prof->pop();
    }

template<class evaluator>
void PotentialExternal<evaluator>::validateType(unsigned int type, std::string action)
    {
    if (type >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid type encountered when " + action);
        }
    }

//! Set the parameters for this potential
/*! \param type type for which to set parameters
    \param params value of parameters
*/
template<class evaluator>
void PotentialExternal<evaluator>::setParams(unsigned int type, const param_type& params)
    {
    validateType(type, std::string("setting parameters in PotentialExternal"));
    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = params;
    }

template<class evaluator> pybind11::object PotentialExternal<evaluator>::getParams(std::string type)
    {
    auto typ = m_pdata->getTypeByName(type);
    validateType(typ, std::string("getting parameters in PotentialExternal"));

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);
    return h_params.data[typ].toPython();
    }

template<class evaluator>
void PotentialExternal<evaluator>::setParamsPython(std::string typ, pybind11::object params)
    {
    unsigned int type_idx = m_pdata->getTypeByName(typ);
    setParams(type_idx, param_type(params));
    }

template<class evaluator> void PotentialExternal<evaluator>::setField(field_type field)
    {
    ArrayHandle<field_type> h_field(m_field, access_location::host, access_mode::overwrite);
    *(h_field.data) = field;
    }

namespace detail
    {
//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternal class template.
*/
template<class T> void export_PotentialExternal(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, ForceCompute, std::shared_ptr<T>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &T::setParamsPython)
        .def("getParams", &T::getParams)
        .def("setField", &T::setField);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
