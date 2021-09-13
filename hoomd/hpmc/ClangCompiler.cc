#include "ClangCompiler.h"

#include <llvm/Support/TargetSelect.h>
#include <llvm/PassRegistry.h>
#include <llvm/InitializePasses.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/CodeGen/CodeGenAction.h>

#include <sstream>
#include <iostream>

std::shared_ptr<ClangCompiler> ClangCompiler::m_clang_compiler = nullptr;

/** Returns a shared pointer to the clang compiler singleton instance
*/
std::shared_ptr<ClangCompiler> ClangCompiler::createClangCompiler()
    {
    if (!m_clang_compiler)
        {
        m_clang_compiler = std::shared_ptr<ClangCompiler>(new ClangCompiler());
        }

    return m_clang_compiler;
    }

/** Initialize the LLVM library
*/
ClangCompiler::ClangCompiler()
    {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto& Registry = *llvm::PassRegistry::getPassRegistry();

    llvm::initializeCore(Registry);
    llvm::initializeScalarOpts(Registry);
    llvm::initializeVectorization(Registry);
    llvm::initializeIPO(Registry);
    llvm::initializeAnalysis(Registry);
    llvm::initializeTransformUtils(Registry);
    llvm::initializeInstCombine(Registry);
    llvm::initializeInstrumentation(Registry);
    llvm::initializeTarget(Registry);
    }

/** @param code The C++ code to compile.
    @param user_args The arguments to pass to the compiler.

    @returns The LLVM IR of the compiled code.
*/
std::string ClangCompiler::compileCode(const std::string& code, const std::vector<std::string>& user_args)
    {
    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options = new clang::DiagnosticOptions();
    // change outs to get text errors in a stringstream? https://clang.llvm.org/doxygen/classclang_1_1TextDiagnosticPrinter.html#af231b7c17ff249332b9399b62552cebe
    clang::TextDiagnosticPrinter diagnostic_printer(llvm::outs(), diagnostic_options.get());
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids = new clang::DiagnosticIDs();

    // the false in the argument list prevents the diagnostic engine from freeing the diagnostic printer
    clang::DiagnosticsEngine diagnostics_engine(diag_ids, diagnostic_options, &diagnostic_printer, false);

    clang::CompilerInstance compiler_instance;
    auto& compiler_invocation = compiler_instance.getInvocation();

    // build up argument list
    std::vector<std::string> clang_args;

    std::stringstream ss;
    ss << "-triple=" << llvm::sys::getDefaultTargetTriple();
    clang_args.push_back(ss.str());
    clang_args.insert(clang_args.end(), user_args.begin(), user_args.end());

    std::vector<const char *> clang_arg_c_strings;
    for (auto& arg : clang_args)
        {
        std::cout << arg << std::endl;
        clang_arg_c_strings.push_back(arg.c_str());
        }

    bool result = clang::CompilerInvocation::CreateFromArgs(compiler_invocation, llvm::ArrayRef<const char*>(clang_arg_c_strings.data(), clang_arg_c_strings.size()), diagnostics_engine);
    if (!result)
        {
        std::cout << "Error creating CompilerInvocation" << std::endl;
        // TODO: handle error condition
        }

//     auto* language_options = compiler_invocation.getLangOpts();
//     auto& preprocessor_options = compiler_invocation.getPreprocessorOpts();
    auto& target_options = compiler_invocation.getTargetOpts();
    auto& frontend_options = compiler_invocation.getFrontendOpts();
    auto& header_search_options = compiler_invocation.getHeaderSearchOpts();
// #ifdef NV_LLVM_VERBOSE
     header_search_options.Verbose = true;
// #endif
//     auto& codeGen_options = compiler_invocation.getCodeGenOpts();

    frontend_options.Inputs.clear();
    frontend_options.Inputs.push_back(clang::FrontendInputFile(llvm::MemoryBufferRef(llvm::StringRef(code), "code"), clang::InputKind(clang::Language::CXX)));

    target_options.Triple = llvm::sys::getDefaultTargetTriple();
    compiler_instance.createDiagnostics(&diagnostic_printer, false);

    llvm::LLVMContext context;
    std::unique_ptr<clang::CodeGenAction> action = std::make_unique<clang::EmitLLVMOnlyAction>(&context);

    if (!compiler_instance.ExecuteAction(*action))
    {
        std::cout << "Error generating code" << std::endl;
    }

    std::unique_ptr<llvm::Module> module = action->takeModule();
    if (!module)
    {
        std::cout << "Error taking module" << std::endl;
    }

    // TODO: Run optimization passes
    // see the posts following this one:
    // https://wiki.nervtech.org/doku.php?id=blog:2020:0410_dynamic_cpp_compilation

//     llvm::PassBuilder passBuilder;
//     llvm::LoopAnalysisManager loopAnalysisManager(codeGenOptions.DebugPassManager);
//     llvm::FunctionAnalysisManager functionAnalysisManager(codeGenOptions.DebugPassManager);
//     llvm::CGSCCAnalysisManager cGSCCAnalysisManager(codeGenOptions.DebugPassManager);
//     llvm::ModuleAnalysisManager moduleAnalysisManager(codeGenOptions.DebugPassManager);

//     passBuilder.registerModuleAnalyses(moduleAnalysisManager);
//     passBuilder.registerCGSCCAnalyses(cGSCCAnalysisManager);
//     passBuilder.registerFunctionAnalyses(functionAnalysisManager);
//     passBuilder.registerLoopAnalyses(loopAnalysisManager);
//     passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager, cGSCCAnalysisManager, moduleAnalysisManager);

//     llvm::ModulePassManager modulePassManager = passBuilder.buildPerModuleDefaultPipeline(llvm::PassBuilder::OptimizationLevel::O3);
//     modulePassManager.run(*module, moduleAnalysisManager);
    }
