#pragma once

// do not include python headers
#define HOOMD_LLVMJIT_BUILD
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "KaleidoscopeJIT.h"

namespace hoomd
    {
// Forward declare box class
struct BoxDim;

namespace hpmc
    {
class ExternalFieldEvalFactory
    {
    public:
    typedef float (*ExternalFieldEvalFnPtr)(const BoxDim& box,
                                            unsigned int type,
                                            const vec3<Scalar>& r_i,
                                            const quat<Scalar>& q_i,
                                            Scalar diameter,
                                            Scalar charge);

    //! Constructor
    ExternalFieldEvalFactory(const std::string& cpp_code,
                             const std::vector<std::string>& compiler_args);

    //! Return the evaluator
    ExternalFieldEvalFnPtr getEval()
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
    ExternalFieldEvalFnPtr m_eval;                     //!< Function pointer to evaluator

    std::string m_error_msg; //!< The error message if initialization fails
    };

    } // end namespace hpmc
    } // end namespace hoomd
