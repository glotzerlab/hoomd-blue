// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "EvalFactory.h"
#include "ClangCompiler.h"

#include <memory>
#include <sstream>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/OrcABISupport.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#pragma GCC diagnostic pop

namespace hoomd
    {
namespace hpmc
    {
//! C'tor
EvalFactory::EvalFactory(const std::string& cpp_code,
                         const std::vector<std::string>& compiler_args,
                         bool is_union)
    {
    std::ostringstream sstream;
    m_eval = nullptr;
    m_alpha = nullptr;
    m_alpha_union = nullptr;

    // initialize LLVM
    auto clang_compiler = ClangCompiler::getClangCompiler();

    // Add the program's symbols into the JIT's search space.
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr))
        {
        m_error_msg = "Error loading program symbols.\n";
        return;
        }

    llvm::LLVMContext Context;

    // compile the module
    auto module = clang_compiler->compileCode(cpp_code, compiler_args, Context, sstream);

    if (!module)
        {
        // if the module didn't load, report an error
        m_error_msg = sstream.str();
        return;
        }

    // Build the JIT
    m_jit = llvm::orc::KaleidoscopeJIT::Create();

    if (!m_jit)
        {
        m_error_msg = "Could not initialize JIT.";
        return;
        }

    // Add the module.
    if (auto E = m_jit->addModule(std::move(module)))
        {
        m_error_msg = "Could not add JIT module.";
        return;
        }

    // Look up the eval function pointer.
    auto eval = m_jit->findSymbol("eval");

    if (!eval)
        {
        m_error_msg = "Could not find eval function in LLVM module.";
        return;
        }

    // Look up the param_array arrays
    if (is_union)
        {
        auto alpha = m_jit->findSymbol("param_array_isotropic");
        if (!alpha)
            {
            m_error_msg = "Could not find param_array_isotropic array in LLVM module.";
            return;
            }

        // also get param_array_constituent if this is a union
        auto alpha_union = m_jit->findSymbol("param_array_constituent");
        if (!alpha_union)
            {
            m_error_msg = "Could not find param_array_constituent array in LLVM module.";
            return;
            }

        /// these casts are like this because 1) it works correctly like this and
        /// 2) trying to use static_cast or reinterpret_cast gives compilation errors
        m_alpha = (float**)(alpha->getAddress());
        m_alpha_union = (float**)(alpha_union->getAddress());
        }
    else // not a union class
        {
        auto alpha = m_jit->findSymbol("param_array");
        if (!alpha)
            {
            m_error_msg = "Could not find param_array array in LLVM module.";
            return;
            }
        /// this cast is like this because 1) it works correctly like this and
        /// 2) trying to use static_cast or reinterpret_cast gives compilation errors
        m_alpha = (float**)(alpha->getAddress());
        }

    /// this cast is like this because 1) it works correctly like this and
    /// 2) trying to use static_cast or reinterpret_cast gives compilation errors
    m_eval = (EvalFnPtr)(long unsigned int)(eval->getAddress());
    }

    } // end namespace hpmc
    } // end namespace hoomd
