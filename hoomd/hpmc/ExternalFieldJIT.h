// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef _EXTERNAL_FIELD_ENERGY_JIT_H_
#define _EXTERNAL_FIELD_ENERGY_JIT_H_

#include "hoomd/BoxDim.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/hpmc/ExternalField.h"

#include "ExternalFieldEvalFactory.h"

#define EXTERNAL_FIELD_JIT_LOG_NAME "jit_energy"

namespace hoomd
    {
namespace hpmc
    {
//! Evaluate external field forces via runtime generated code
/*! This class enables the widest possible use-cases of external fields in HPMC with low energy
   barriers for users to add custom forces that execute with high performance. It provides a generic
   interface for returning the energy of interaction between a particle and an external field. The
   actual computation is performed by code that is loaded and compiled at run time using LLVM.

    The user provides C++ code containing a function 'eval' with the defined function signature.
   On construction, this class uses the LLVM library to compile that down to machine code and
   obtain a function pointer to call.

    LLVM execution is managed with the KaleidoscopeJIT class in m_JIT. On construction, the LLVM
   module is loaded and compiled. KaleidoscopeJIT handles construction of C++ static members,
   etc.... When m_JIT is deleted, all of the compiled code and memory used in the module is deleted.
   KaleidoscopeJIT takes care of destructing C++ static members inside the module.

    LLVM JIT is capable of calling any function in the hosts address space. ExternalFieldJIT does
   not take advantage of that, limiting the user to a very specific API for computing the energy
   between a pair of particles.
*/
template<class Shape> class ExternalFieldJIT : public hpmc::ExternalFieldMono<Shape>
    {
    public:
    //! Constructor
    ExternalFieldJIT(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ExecutionConfiguration> exec_conf,
                     const std::string& cpu_code,
                     const std::vector<std::string>& compiler_args,
                     pybind11::array_t<float> param_array)
        : hpmc::ExternalFieldMono<Shape>(sysdef), m_exec_conf(exec_conf),
          m_param_array(param_array.data(),
                        param_array.data() + param_array.size(),
                        hoomd::detail::managed_allocator<float>(m_exec_conf->isCUDAEnabled()))
        {
        // build the JIT.
        ExternalFieldEvalFactory* factory = new ExternalFieldEvalFactory(cpu_code, compiler_args);

        // get the evaluator
        m_eval = factory->getEval();

        if (!m_eval)
            {
            throw std::runtime_error("Error compiling JIT code for CPPExternalPotential.\n"
                                     + factory->getError());
            }
        factory->setAlphaArray(&m_param_array.front());
        m_factory = std::shared_ptr<ExternalFieldEvalFactory>(factory);
        }

    float energy_no_wrap(const BoxDim& box,
                         unsigned int type,
                         const vec3<Scalar>& r_i,
                         const quat<Scalar>& q_i,
                         Scalar diameter,
                         Scalar charge)
        {
        return m_eval(box, type, r_i, q_i, diameter, charge);
        }

    //! Evaluate the energy of the force.
    /*! \param box The system box.
        \param type Particle type.
        \param r_i Particle position
        \param q_i Particle orientation.
        \param diameter Particle diameter.
        \param charge Particle charge.
        \returns Energy due to the force
    */
    virtual float energy(const BoxDim& box,
                         unsigned int type,
                         const vec3<Scalar>& r_i,
                         const quat<Scalar>& q_i,
                         Scalar diameter,
                         Scalar charge)
        {
        vec3<Scalar> r_i_wrapped = r_i - vec3<Scalar>(this->m_pdata->getOrigin());
        int3 image = make_int3(0, 0, 0);
        box.wrap(r_i_wrapped, image);
        return energy_no_wrap(box, type, r_i_wrapped, q_i, diameter, charge);
        }

    //! Computes the total field energy of the system at the current state
    virtual double computeEnergy(uint64_t timestep)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                     access_location::host,
                                     access_mode::read);

        const BoxDim box = this->m_pdata->getGlobalBox();

        double total_energy = 0.0;
        for (size_t i = 0; i < this->m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            unsigned int typ_i = __scalar_as_int(postype_i.w);

            total_energy += energy(box,
                                   typ_i,
                                   vec3<Scalar>(postype_i),
                                   quat<Scalar>(h_orientation.data[i]),
                                   h_diameter.data[i],
                                   h_charge.data[i]);
            }
#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &total_energy,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          this->m_exec_conf->getMPICommunicator());
            }
#endif
        return total_energy;
        }

    virtual double calculateDeltaE(uint64_t timestep,
                                   const Scalar4* const position_old_arg,
                                   const Scalar4* const orientation_old_arg,
                                   const BoxDim& box_old,
                                   const Scalar3& origin_old)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar4> h_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                     access_location::host,
                                     access_mode::read);
        const BoxDim box_new = this->m_pdata->getGlobalBox();
        const Scalar4 *position_old = position_old_arg, *orientation_old = orientation_old_arg;
        if (!position_old)
            {
            const Scalar4* const position_new = h_postype.data;
            position_old = position_new;
            }
        if (!orientation_old)
            {
            const Scalar4* const orientation_new = h_orientation.data;
            orientation_old = orientation_new;
            }
        double dE = 0.0;
        for (size_t i = 0; i < this->m_pdata->getN(); i++)
            {
            // read in the current position and orientation
            Scalar4 postype_i = h_postype.data[i];
            unsigned int typ_i = __scalar_as_int(postype_i.w);
            int3 image = make_int3(0, 0, 0);
            vec3<Scalar> old_pos_i = vec3<Scalar>(*(position_old + i)) - vec3<Scalar>(origin_old);
            box_old.wrap(old_pos_i, image);
            dE += energy(box_new,
                         typ_i,
                         vec3<Scalar>(postype_i),
                         quat<Scalar>(h_orientation.data[i]),
                         h_diameter.data[i],
                         h_charge.data[i]);
            dE -= energy_no_wrap(box_old,
                                 typ_i,
                                 old_pos_i,
                                 quat<Scalar>(*(orientation_old + i)),
                                 h_diameter.data[i],
                                 h_charge.data[i]);
            }
#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &dE,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          this->m_exec_conf->getMPICommunicator());
            }
#endif
        return dE;
        }

    //! method to calculate the energy difference for the proposed move.
    double energydiff(uint64_t timestep,
                      const unsigned int& index,
                      const vec3<Scalar>& position_old,
                      const Shape& shape_old,
                      const vec3<Scalar>& position_new,
                      const Shape& shape_new)
        {
        ArrayHandle<Scalar4> h_postype(this->m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        Scalar4 postype_i = h_postype.data[index];
        unsigned int typ_i = __scalar_as_int(postype_i.w);

        ArrayHandle<Scalar> h_diameter(this->m_pdata->getDiameters(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<Scalar> h_charge(this->m_pdata->getCharges(),
                                     access_location::host,
                                     access_mode::read);

        double dE = 0.0;
        dE += energy(this->m_pdata->getGlobalBox(),
                     typ_i,
                     position_new,
                     shape_new.orientation,
                     h_diameter.data[index],
                     h_charge.data[index]);
        dE -= energy(this->m_pdata->getGlobalBox(),
                     typ_i,
                     position_old,
                     shape_old.orientation,
                     h_diameter.data[index],
                     h_charge.data[index]);
        return dE;
        }

    static pybind11::object getParamArray(pybind11::object self)
        {
        auto self_cpp = self.cast<ExternalFieldJIT*>();
        return pybind11::array(self_cpp->m_param_array.size(),
                               self_cpp->m_factory->getAlphaArray(),
                               self);
        }

    protected:
    std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
    //! function pointer signature
    typedef float (*ExternalFieldEvalFnPtr)(const BoxDim& box,
                                            unsigned int type,
                                            const vec3<Scalar>& r_i,
                                            const quat<Scalar>& q_i,
                                            Scalar diameter,
                                            Scalar charge);
    std::shared_ptr<ExternalFieldEvalFactory> m_factory; //!< The factory for the evaluator function
    ExternalFieldEvalFactory::ExternalFieldEvalFnPtr
        m_eval; //!< Pointer to evaluator function inside the JIT module
    std::vector<float, hoomd::detail::managed_allocator<float>>
        m_param_array; //!< array containing adjustable parameters
    };

//! Exports the ExternalFieldJIT class to python
template<class Shape> void export_ExternalFieldJIT(pybind11::module& m, std::string name)
    {
    pybind11::class_<ExternalFieldJIT<Shape>,
                     hpmc::ExternalFieldMono<Shape>,
                     std::shared_ptr<ExternalFieldJIT<Shape>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExecutionConfiguration>,
                            const std::string&,
                            const std::vector<std::string>&,
                            pybind11::array_t<float>>())
        .def("computeEnergy", &ExternalFieldJIT<Shape>::computeEnergy)
        .def_property_readonly("param_array", &ExternalFieldJIT<Shape>::getParamArray);
    }

    } // end namespace hpmc
    } // end namespace hoomd
#endif // _EXTERNAL_FIELD_ENERGY_JIT_H_
