#ifndef _EXTERNAL_FIELD_ENERGY_JIT_H_
#define _EXTERNAL_FIELD_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include "ExternalFieldEvalFactory.h"

#define EXTERNAL_FIELD_ENERGY_LOG_NAME           "force_energy"

//! Evaluate external field forces via runtime generated code
/*! This class enables the widest possible use-cases of external fields in HPMC with low energy barriers for users to add
    custom forces that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a particle and an external field. The actual computation is performed by code that is loaded and
    compiled at run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    LLVM execution is managed with the KaleidoscopeJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. KaleidoscopeJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. KaleidoscopeJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. ExternalFieldJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
class ExternalFieldJIT : public hpmc::ForceEnergy
    {
    public:
        //! Constructor
        ExternalFieldJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir);

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
            Scalar charge
            )
            {
            return m_eval(box, type, r_i, q_i, diameter, charge);
            }

    protected:
        //! function pointer signature
        typedef float (*ExternalFieldEvalFnPtr)(const BoxDim& box, unsigned int type, const vec3<Scalar>& r_i, const quat<Scalar>& q_i, Scalar diameter, Scalar charge);
        std::shared_ptr<ExternalFieldEvalFactory> m_factory;       //!< The factory for the evaluator function
        ExternalFieldEvalFactory::ExternalFieldEvalFnPtr m_eval;                //!< Pointer to evaluator function inside the JIT module
    };

//! Exports the ExternalFieldJIT class to python
void export_ExternalFieldJIT(pybind11::module &m);
#endif // _EXTERNAL_FIELD_ENERGY_JIT_H_
