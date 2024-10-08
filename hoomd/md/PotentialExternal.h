// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/VectorMath.h"
#include "hoomd/managed_allocator.h"
#include "hoomd/md/EvaluatorExternalPeriodic.h"
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
 *
 * Note: A field_type of void* for the evaluator template type indicates that no field_type actually
 * exists. Some type is needed for code to compile.
 */
template<class evaluator> class PotentialExternal : public ForceCompute
    {
    public:
    //! Constructs the compute
    PotentialExternal(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~PotentialExternal();

    //! type of external potential parameters
    typedef typename evaluator::param_type param_type;
    typedef typename evaluator::field_type field_type;

    bool isAnisotropic();

    //! Sets parameters of the evaluator
    pybind11::object getParams(std::string type);

    //! set the potential parameters via cpp arguments
    void setParams(unsigned int type, const param_type& params);

    //! set the potential parameters via python arguments
    void setParamsPython(std::string typ, pybind11::object params);

    //! make sure the type index is within range
    void validateType(unsigned int type, std::string action);

    //! set the field type of the evaluator
    void setField(std::shared_ptr<field_type>& field);

    //! get a reference to the field parameters. Used to expose the field attributes to Python.
    std::shared_ptr<field_type>& getField();

    protected:
    GPUArray<param_type> m_params;       //!< Array of per-type parameters
    std::shared_ptr<field_type> m_field; /// evaluator dependent field parameters

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

/*! Constructor
    \param sysdef system definition
*/
template<class evaluator>
PotentialExternal<evaluator>::PotentialExternal(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef),
      m_field(hoomd::detail::make_managed_shared<typename PotentialExternal<evaluator>::field_type>(
          m_exec_conf->isCUDAEnabled()))
    {
    GPUArray<param_type> params(m_pdata->getNTypes(), m_exec_conf);
    m_params.swap(params);
    }

/*! Destructor
 */
template<class evaluator> PotentialExternal<evaluator>::~PotentialExternal() { }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator> void PotentialExternal<evaluator>::computeForces(uint64_t timestep)
    {
    if (std::is_same<evaluator, EvaluatorExternalPeriodic>() && m_sysdef->getNDimensions() == 2)
        {
        throw std::runtime_error("The external periodic potential is not valid in 2D boxes.");
        }

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);

    ArrayHandle<param_type> h_params(m_params, access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getGlobalBox();

    unsigned int nparticles = m_pdata->getN();

    // Zero data for force calculation.
    memset((void*)h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_torque.data, 0, sizeof(Scalar4) * m_torque.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_torque.data);
    assert(h_virial.data);

    // for each of the particles
    for (unsigned int idx = 0; idx < nparticles; idx++)
        {
        // get the current particle properties
        Scalar3 X = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);
        quat<Scalar> q(h_orientation.data[idx]);
        Scalar3 F, T;
        Scalar energy;
        Scalar virial[6];

        evaluator eval(X, q, box, h_params.data[type], *m_field);

        if (evaluator::needsCharge())
            {
            Scalar qi = h_charge.data[idx];
            eval.setCharge(qi);
            }
        eval.evalForceTorqueEnergyAndVirial(F, T, energy, virial);

        // apply the constraint force
        h_force.data[idx].x = F.x;
        h_force.data[idx].y = F.y;
        h_force.data[idx].z = F.z;
        h_force.data[idx].w = energy;
        for (int k = 0; k < 6; k++)
            h_virial.data[k * m_virial_pitch + idx] = virial[k];

        h_torque.data[idx].x = T.x;
        h_torque.data[idx].y = T.y;
        h_torque.data[idx].z = T.z;
        }
    }

template<class evaluator>
void PotentialExternal<evaluator>::validateType(unsigned int type, std::string action)
    {
    if (type >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid type encountered when " + action);
        }
    }

//! Returns true if this ForceCompute requires anisotropic integration
template<class evaluator> bool PotentialExternal<evaluator>::isAnisotropic()
    {
    // by default, only translational degrees of freedom are integrated
    return evaluator::isAnisotropic();
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

template<class evaluator>
void PotentialExternal<evaluator>::setField(
    std::shared_ptr<PotentialExternal<evaluator>::field_type>& field)
    {
    m_field = field;
    }

template<class evaluator>
std::shared_ptr<typename PotentialExternal<evaluator>::field_type>&
PotentialExternal<evaluator>::getField()
    {
    return m_field;
    }

namespace detail
    {
//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialExternal(pybind11::module& m, const std::string& name)
    {
    auto cls = pybind11::class_<PotentialExternal<T>,
                                ForceCompute,
                                std::shared_ptr<PotentialExternal<T>>>(m, name.c_str())
                   .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
                   .def("setParams", &PotentialExternal<T>::setParamsPython)
                   .def("getParams", &PotentialExternal<T>::getParams);

    // void* serves as a sentinel type indicating that no field_type actually exists.
    if constexpr (!std::is_same<typename T::field_type, void*>::value)
        {
        cls.def_property("field", &PotentialExternal<T>::getField, &PotentialExternal<T>::setField);
        }
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
#endif
