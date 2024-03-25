// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

// do not include python headers
#define HOOMD_LLVMJIT_BUILD
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#include "KaleidoscopeJIT.h"

namespace hoomd
    {
namespace hpmc
    {
class EvalFactory
    {
    public:
    typedef float (*EvalFnPtr)(const vec3<float>& r_ij,
                               unsigned int type_i,
                               const quat<float>& q_i,
                               float d_i,
                               float charge_i,
                               unsigned int type_j,
                               const quat<float>& q_j,
                               float d_j,
                               float charge_j);

    //! Constructor
    EvalFactory(const std::string& cpp_code,
                const std::vector<std::string>& compiler_args,
                bool is_union);

    //! Return the evaluator
    EvalFnPtr getEval()
        {
        return m_eval;
        }

    //! Get the error message from initialization
    const std::string& getError()
        {
        return m_error_msg;
        }

    //! Retrieve alpha array
    float* getAlphaArray() const
        {
        return *m_alpha;
        }

    //! Retrieve alpha array
    void setAlphaArray(float* h_alpha)
        {
        *m_alpha = h_alpha;
        }

    //! Retrieve alpha array
    float* getAlphaUnionArray()
        {
        return *m_alpha_union;
        }

    //! Set alpha union array
    void setAlphaUnionArray(float* h_alpha_union)
        {
        *m_alpha_union = h_alpha_union;
        }

    private:
    std::unique_ptr<llvm::orc::KaleidoscopeJIT> m_jit; //!< The persistent JIT engine
    EvalFnPtr m_eval;                                  //!< Function pointer to evaluator
    float** m_alpha;                                   // Pointer to alpha array
    float** m_alpha_union;                             // Pointer to alpha array for union
    std::string m_error_msg; //!< The error message if initialization fails
    };

    } // end namespace hpmc
    } // end namespace hoomd
