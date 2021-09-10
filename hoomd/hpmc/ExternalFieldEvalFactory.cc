#include "ExternalFieldEvalFactory.h"
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

#include "llvm/Support/raw_os_ostream.h"

#pragma GCC diagnostic pop

//! C'tor
ExternalFieldEvalFactory::ExternalFieldEvalFactory(const std::string& llvm_ir)
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

    llvm::LLVMContext Context;
    llvm::SMDiagnostic Err;

    // Read the input IR data
    llvm::StringRef ir_str(llvm_ir);
    std::unique_ptr<llvm::MemoryBuffer> ir_membuf = llvm::MemoryBuffer::getMemBuffer(ir_str);
    std::unique_ptr<llvm::Module> module = llvm::parseIR(*ir_membuf, Err, Context);

    if (!module)
        {
        // if the module didn't load, report an error
        Err.print("ExternalFieldEvalFactory", llvm_err);
        llvm_err.flush();
        m_error_msg = sstream.str();
        return;
        }

    // Build the JIT
    m_jit = llvm::orc::KaleidoscopeJIT::Create();

    if (!m_jit)
        {
        m_error_msg = "Could not initialize JIT.\n";
        return;
        }

    // Add the module.
    if (auto E = m_jit->addModule(std::move(module)))
        {
        m_error_msg = "Could not add JIT module.\n";
        return;
        }

    // Look up the eval function pointer.
    auto eval = m_jit->findSymbol("eval");

    if (!eval)
        {
        m_error_msg = "Could not find eval function in LLVM module.\n";
        return;
        }

    m_eval = (ExternalFieldEvalFnPtr)(long unsigned int)(eval->getAddress());

    llvm_err.flush();
    }
