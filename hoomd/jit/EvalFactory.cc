#include <utility>
#include <memory>
#include <sstream>
#include "EvalFactory.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 9)
#include "llvm/ExecutionEngine/Orc/OrcABISupport.h"
#else
#include "llvm/ExecutionEngine/Orc/OrcArchitectureSupport.h"
#endif
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/DynamicLibrary.h"

#include "llvm/Support/raw_os_ostream.h"

//! C'tor
EvalFactory::EvalFactory(const std::string& llvm_ir)
    {
    // set to null pointer
    m_eval = NULL;

    // initialize LLVM
    std::ostringstream sstream;
    llvm::raw_os_ostream llvm_err(sstream);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Add the program's symbols into the JIT's search space.
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr))
        {
            m_error_msg = "Error loading program symbols.\n";
            return;
        }

    #if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR > 3 || (LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 9)
    llvm::LLVMContext Context;
    #else
    llvm::LLVMContext &Context = llvm::getGlobalContext();
    #endif
    llvm::SMDiagnostic Err;

    // Read the input IR data
    llvm::StringRef ir_str(llvm_ir);
    std::unique_ptr<llvm::MemoryBuffer> ir_membuf = llvm::MemoryBuffer::getMemBuffer(ir_str);
    std::unique_ptr<llvm::Module> Mod = llvm::parseIR(*ir_membuf, Err, Context);

    if (!Mod)
        {
        // if the module didn't load, report an error
        Err.print("EvalFactory", llvm_err);
        llvm_err.flush();
        m_error_msg = sstream.str();
        return;
        }

    // Build the JIT
    m_jit = std::unique_ptr<llvm::orc::KaleidoscopeJIT>(new llvm::orc::KaleidoscopeJIT());

    // Add the module, look up main and run it.
    m_jit->addModule(std::move(Mod));

    auto eval = m_jit->findSymbol("eval");

    if (!eval)
        {
        m_error_msg = "Could not find eval function in LLVM module.\n";
        return;
        }

    auto alpha = m_jit->findSymbol("alpha_iso");

    if (!alpha)
        {
        m_error_msg = "Could not find alpha array in LLVM module.\n";
        return;
        }

    auto alpha_union = m_jit->findSymbol("alpha_union");

    if (!alpha_union)
        {
        m_error_msg = "Could not find alpha_union array in LLVM module.\n";
        return;
        }

    #if defined LLVM_VERSION_MAJOR && LLVM_VERSION_MAJOR >= 5
    m_eval = (EvalFnPtr)(long unsigned int)(cantFail(eval.getAddress()));
    m_alpha = (float *)(cantFail(alpha.getAddress()));
    m_alpha_union = (float *)(cantFail(alpha_union.getAddress()));
    #else
    m_eval = (EvalFnPtr) eval.getAddress();
    m_alpha = (float *) alpha.getAddress();
    m_alpha_union = (float *) alpha_union.getAddress();
    #endif

    llvm_err.flush();
    }
