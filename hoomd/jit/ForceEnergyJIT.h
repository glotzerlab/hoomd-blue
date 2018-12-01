#ifndef _FORCE_ENERGY_JIT_H_
#define _FORCE_ENERGY_JIT_H_

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/ExecutionConfiguration.h"
#include "hoomd/hpmc/IntegratorHPMC.h"

#include "EvalFactory.h"

#define FORCE_ENERGY_LOG_NAME           "patch_energy"
#define FORCE_ENERGY_RCUT               "patch_energy_rcut"

//! Evaluate external field forces via runtime generated code
/*! This class enables the widest possible use-cases of external fields in HPMC with low energy barriers for users to add
    custom forces that execute with high performance. It provides a generic interface for returning the energy of
    interaction between a particle and an external field. The actual computation is performed by code that is loaded and
    compiled at run time using LLVM.

    The user provides LLVM IR code containing a function 'eval' with the defined function signature. On construction,
    this class uses the LLVM library to compile that IR down to machine code and obtain a function pointer to call.

    LLVM execution is managed with the OrcLazyJIT class in m_JIT. On construction, the LLVM module is loaded and
    compiled. OrcLazyJIT handles construction of C++ static members, etc.... When m_JIT is deleted, all of the compiled
    code and memory used in the module is deleted. OrcLazyJIT takes care of destructing C++ static members inside the
    module.

    LLVM JIT is capable of calling any function in the hosts address space. ForceEnergyJIT does not take advantage of
    that, limiting the user to a very specific API for computing the energy between a pair of particles.
*/
class ForceEnergyJIT : public hpmc::ForceEnergy
    {
    public:
        //! Constructor
        ForceEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir, Scalar r_cut);

        //! Evaluate the energy of the force.
        /*! \param box The system box.
            \param type Particle type.
            \param pos Particle position
            \param orientation Particle orientation.
            \param diameter Particle diameter.
            \param charge Particle charge.
            \returns Energy due to the force
        */
        virtual float eval(const boxDim& box,
            unsigned int type,
            vec3<Scalar> pos,
            Scalar4 orientation
            Scalar diameter,
            Scalar charge
            )
            {
            return m_eval(const boxDim& box, unsigned int type, vec3<Scalar> pos, Scalar4 orientation Scalar diameter, Scalar charge)
            }

    protected:
        //! function pointer signature
        typedef float (*EvalFnPtr)(const boxDim& box, unsigned int type, vec3<Scalar> pos, Scalar4 orientation Scalar diameter, Scalar charge);
        std::shared_ptr<EvalFactory> m_factory;       //!< The factory for the evaluator function
        EvalFactory::EvalFnPtr m_eval;                //!< Pointer to evaluator function inside the JIT module
    };

//! Exports the ForceEnergyJIT class to python
void export_ForceEnergyJIT(pybind11::module &m);
#endif // _FORCE_ENERGY_JIT_H_
