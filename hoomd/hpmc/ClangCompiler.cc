// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ClangCompiler.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/Version.inc>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/HeaderSearch.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/InitializePasses.h>
#include <llvm/PassRegistry.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_os_ostream.h>

#pragma GCC diagnostic pop

#include <iostream>
#include <sstream>

namespace hoomd
    {
namespace hpmc
    {
std::shared_ptr<ClangCompiler> ClangCompiler::m_clang_compiler = nullptr;

/** Returns a shared pointer to the clang compiler singleton instance
 */
std::shared_ptr<ClangCompiler> ClangCompiler::getClangCompiler()
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

    @returns The LLVM module with the code compiled.
*/
std::unique_ptr<llvm::Module> ClangCompiler::compileCode(const std::string& code,
                                                         const std::vector<std::string>& user_args,
                                                         llvm::LLVMContext& context,
                                                         std::ostringstream& out)
    {
    // initialize the diagnostics engine to write compilation warnings/errors to stdout/stderr
    clang::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnostic_options
        = new clang::DiagnosticOptions();
    llvm::raw_os_ostream llvm_out(out);
    clang::TextDiagnosticPrinter diagnostic_printer(llvm_out, diagnostic_options.get());
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids = new clang::DiagnosticIDs();
    clang::DiagnosticsEngine diagnostics_engine(diag_ids,
                                                diagnostic_options,
                                                &diagnostic_printer,
                                                false);

    // determine the clang resource path
    std::string resource_path(HOOMD_LLVM_INSTALL_PREFIX);

    // build up the argument list
    std::vector<std::string> clang_args;
    clang_args.push_back("-D");
    clang_args.push_back("HOOMD_LLVMJIT_BUILD");
    clang_args.push_back("--std=c++14");
    // prevent the driver from creating empty output files in /tmp
    clang_args.push_back("-S");
    clang_args.push_back("-emit-llvm");
    clang_args.insert(clang_args.end(), user_args.begin(), user_args.end());
    clang_args.push_back("_hoomd_llvm_code.cc");

    // convert arguments to a char** array.
    std::vector<const char*> clang_arg_c_strings;
    clang_arg_c_strings.push_back("clang");
    for (auto& arg : clang_args)
        {
        clang_arg_c_strings.push_back(arg.c_str());
        }

    // This is a way to get system wide compiler options from the clang Driver interface.
    // It is needed to find the standard library include directories.
    // https://cpp.hotexamples.com/site/file?hash=0xd4e048edbee7a77d7b2181909e61ab1a1213629fe8aa79248fea8ae17f8dc7fc&fullName=safecode-mirror-master/tools/clang/examples/main.cpp&project=lygstate/safecode-mirror
    // see also:
    // https://cpp.hotexamples.com/examples/-/CompilerInstance/-/cpp-compilerinstance-class-examples.html
    std::cout << "################################################" << std::endl;
    std::cout << "##  ###  ###     ###  #####  #####        ##  ##" << std::endl;
    std::cout << "##  ###  ###  ######  #####  #####  ####  ##  ##" << std::endl;
    std::cout << "##       ###     ###  #####  #####  ####  ##  ##" << std::endl;
    std::cout << "##  ###  ###  ######  #####  #####  ####  ######" << std::endl;
    std::cout << "##  ###  ###     ###     ##     ##        ##  ##" << std::endl;
    std::cout << "################################################" << std::endl;
    std::cout << "HOOMD_LLVM_INSTALL_PREFIX =" << HOOMD_LLVM_INSTALL_PREFIX << std::endl;
    std::cout << "resource_path = " << resource_path << std::endl;
    std::cout << "clang_args = " << std::endl;
    for (int i = 0; i < clang_args.size(); i++)
        {
        std::cout << clang_args[i] << std::endl;
        }
    std::string clang_exec_with_path = resource_path + "/bin/clang";
    std::ifstream f(clang_exec_with_path.c_str());
    if (!f.good())
        {
        throw std::runtime_error("cannot find $PREFIX/bin/clang");
        }

    clang::driver::Driver driver(resource_path + "/bin/clang",
                                 llvm::sys::getDefaultTargetTriple(),
                                 diagnostics_engine);
    driver.setCheckInputsExist(false);
    llvm::ArrayRef<const char*> RF(&(clang_arg_c_strings[0]), clang_arg_c_strings.size());
    auto compilation = driver.BuildCompilation(RF);
    // This may work in future LLVM releases
    // auto CC1Args = clang::tooling::getCC1Arguments(Diagnostics, Compilation.get());
    // if (CC1Args == NULL)
    //     {
    //     return 0;
    //     }
    // use this for now
    auto& cc_args = compilation->getJobs().begin()->getArguments();

    // Write out the compilation arguments for debugging purposes
    out << "Compilation arguments:" << std::endl;
    for (unsigned int i = 0; i < cc_args.size(); i++)
        {
        out << cc_args[i] << std::endl;
        }
    out << std::endl;

    // initialize the compiler instance with the args provided by the driver interface
    clang::CompilerInstance compiler_instance;
    auto& compiler_invocation = compiler_instance.getInvocation();
    bool result = clang::CompilerInvocation::CreateFromArgs(
        compiler_invocation,
        llvm::ArrayRef<const char*>(cc_args.data(), cc_args.size()),
        diagnostics_engine);
    if (!result)
        {
        out << "Error creating CompilerInvocation." << std::endl;
        return nullptr;
        }

    // replace the input file argument with the in memory code
    auto& frontend_options = compiler_invocation.getFrontendOpts();
    frontend_options.Inputs.clear();
#if LLVM_VERSION_MAJOR >= 12
    frontend_options.Inputs.push_back(
        clang::FrontendInputFile(llvm::MemoryBufferRef(llvm::StringRef(code), "code.cc"),
                                 clang::InputKind(clang::Language::CXX)));
#else
    auto code_buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(code), "code.cc");
    frontend_options.Inputs.push_back(
        clang::FrontendInputFile(code_buffer.get(), clang::InputKind(clang::Language::CXX)));
#endif

    // configure output streams for the compiler
    compiler_instance.setVerboseOutputStream(llvm_out);
    compiler_instance.createDiagnostics(&diagnostic_printer, false); // keep this or llvm seg faults

    // generate the code
    std::unique_ptr<clang::CodeGenAction> action
        = std::make_unique<clang::EmitLLVMOnlyAction>(&context);

    if (!compiler_instance.ExecuteAction(*action))
        {
        out << "Error generating code." << std::endl;
        return nullptr;
        }

    // get the module
    std::unique_ptr<llvm::Module> module = action->takeModule();
    if (!module)
        {
        out << "Error taking module." << std::endl;
        return nullptr;
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
    //     passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager,
    //     cGSCCAnalysisManager, moduleAnalysisManager);

    //     llvm::ModulePassManager modulePassManager =
    //     passBuilder.buildPerModuleDefaultPipeline(llvm::PassBuilder::OptimizationLevel::O3);
    //     modulePassManager.run(*module, moduleAnalysisManager);

    return module;
    }

    } // end namespace hpmc
    } // end namespace hoomd
