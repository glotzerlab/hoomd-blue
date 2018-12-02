#pragma once

// do not include python headers
#define HOOMD_NOPYTHON
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "hoomd/BoxDim.h"

#include "KaleidoscopeJIT.h"

class ForceEvalFactory
    {
    public:
        typedef float (*ForceEvalFnPtr)(const BoxDim& box,
            unsigned int type,
            vec3<Scalar> pos,
            Scalar4 orientation,
            Scalar diameter,
            Scalar charge
            );

        //! Constructor
        ForceEvalFactory(const std::string& llvm_ir);

        //! Return the evaluator
        ForceEvalFnPtr getEval()
            {
            return m_eval;
            }

        //! Get the error message from initialization
        const std::string& getError()
            {
            return m_error_msg;
            }

    private:
        std::unique_ptr<llvm::orc::KaleidoscopeJIT> m_jit; //!< The persistent JIT engine
        ForceEvalFnPtr m_eval;         //!< Function pointer to evaluator

        std::string m_error_msg; //!< The error message if initialization fails
    };
