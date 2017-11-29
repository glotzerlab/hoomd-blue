#include <utility>
#include <memory>
#include "EvalFactory.h"
#include "OrcLazyJIT.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/ExecutionEngine/Orc/OrcArchitectureSupport.h"
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

    llvm::LLVMContext &Context = llvm::getGlobalContext();
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

    // Build engine with JIT
    llvm::EngineBuilder EB;
    auto TM = std::unique_ptr<llvm::TargetMachine>(EB.selectTarget());
    auto CompileCallbackMgr = llvm::OrcLazyJIT::createCompileCallbackMgr(llvm::Triple(TM->getTargetTriple()));

    // If we couldn't build the factory function then there must not be a callback
    // manager for this target. Bail out.
    if (!CompileCallbackMgr)
        {
        m_error_msg = "No callback manager available for target '" + TM->getTargetTriple().str() + "'.\n";
        return;
        }

    auto IndirectStubsMgrBuilder = llvm::OrcLazyJIT::createIndirectStubsMgrBuilder(llvm::Triple(TM->getTargetTriple()));

    // If we couldn't build a stubs-manager-builder for this target then bail out.
    if (!IndirectStubsMgrBuilder)
        {
        m_error_msg = "No indirect stubs manager available for target '" + TM->getTargetTriple().str() + "'.\n";
        return;
        }

    // Everything looks good. Build the JIT.
    m_jit = std::shared_ptr<llvm::OrcLazyJIT>(new llvm::OrcLazyJIT(std::move(TM),
                                std::move(CompileCallbackMgr),
                                std::move(IndirectStubsMgrBuilder),
                                true));

    // Add the module, look up main and run it.
    auto MainHandle = m_jit->addModule(std::move(Mod));
    auto eval = m_jit->findSymbolIn(MainHandle, "eval");

    if (!eval)
        {
        m_error_msg = "Could not find eval function in LLVM module.\n";
        return;
        }

    llvm_err.flush();
    m_eval = llvm::fromTargetAddress<EvalFnPtr>(eval.getAddress());
    }
