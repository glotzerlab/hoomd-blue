#include "PatchEnergyJIT.h"

#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/ExecutionEngine/Orc/OrcArchitectureSupport.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Support/DynamicLibrary.h"

#include "llvm/Support/raw_os_ostream.h"

#include <sstream>

#define PATCH_ENERGY_LOG_NAME           "patch_energy"
#define PATCH_ENERGY_RCUT               "patch_energy_rcut"
/*! \param exec_conf The execution configuration (used for messages and MPI communication)
    \param fname File name of the LLVM IR to load
    \param r_cut Center to center distance beyond which the patch energy is 0

    After construction, the LLVM IR is loaded, compiled, and the energy() method is ready to be called.
*/
PatchEnergyJIT::PatchEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& fname, Scalar r_cut) : m_r_cut(r_cut)
    {

    //m_PatchProvidedQuantities.push_back(PATCH_ENERGY_LOG_NAME);
    //m_PatchProvidedQuantities.push_back(PATCH_ENERGY_RCUT);

    // initialize LLVM
    std::ostringstream sstream;
    llvm::raw_os_ostream llvm_err(sstream);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Add the program's symbols into the JIT's search space.
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr))
        {
        exec_conf->msg->error() << "Error loading program symbols.\n";
        throw std::runtime_error("Error loading program symbols.");
        }

    llvm::LLVMContext &Context = llvm::getGlobalContext();
    llvm::SMDiagnostic Err;

    // Read the input file
    std::unique_ptr<llvm::Module> Mod = llvm::parseIRFile(fname, Err, Context);

    if (!Mod)
        {
        // if the module didn't load, report an error
        exec_conf->msg->error() << "could not load LLVM IR" << std::endl;
        Err.print("PatchEnergyJIT", llvm_err);
        llvm_err.flush();
        exec_conf->msg->errorStr(sstream.str());
        throw std::runtime_error("could not load LLVM IR");
        }

    // Build engine with JIT
    llvm::EngineBuilder EB;
    auto TM = std::unique_ptr<llvm::TargetMachine>(EB.selectTarget());
    auto CompileCallbackMgr = llvm::OrcLazyJIT::createCompileCallbackMgr(llvm::Triple(TM->getTargetTriple()));

    // If we couldn't build the factory function then there must not be a callback
    // manager for this target. Bail out.
    if (!CompileCallbackMgr)
        {
        exec_conf->msg->error() << "No callback manager available for target '"
                                << TM->getTargetTriple().str() << "'.\n";
        throw std::runtime_error("Error compiling LLVM JIT.");
        }

    auto IndirectStubsMgrBuilder = llvm::OrcLazyJIT::createIndirectStubsMgrBuilder(llvm::Triple(TM->getTargetTriple()));

    // If we couldn't build a stubs-manager-builder for this target then bail out.
    if (!IndirectStubsMgrBuilder)
        {
        exec_conf->msg->error() << "No indirect stubs manager available for target '"
                                << TM->getTargetTriple().str() << "'.\n";
        throw std::runtime_error("Error compiling LLVM JIT.");
        }

    // Everything looks good. Build the JIT.
    m_JIT =  std::shared_ptr<llvm::OrcLazyJIT>(new llvm::OrcLazyJIT(std::move(TM),
                                                                    std::move(CompileCallbackMgr),
                                                                    std::move(IndirectStubsMgrBuilder),
                                                                    true));

    // Add the module, look up main and run it.
    auto MainHandle = m_JIT->addModule(std::move(Mod));
    auto eval = m_JIT->findSymbolIn(MainHandle, "eval");

    if (!eval)
        {
        exec_conf->msg->error() << "Could not find eval function in LLVM module.\n";
        throw std::runtime_error("Error running LLVM JIT.");
        }

    m_eval = llvm::fromTargetAddress<EvalFnPtr>(eval.getAddress());
    llvm_err.flush();

    }




void export_PatchEnergyJIT(pybind11::module &m)
    {
      pybind11::class_<hpmc::PatchEnergy, std::shared_ptr<hpmc::PatchEnergy> >(m, "PatchEnergy")
              .def(pybind11::init< >());
    pybind11::class_<PatchEnergyJIT, std::shared_ptr<PatchEnergyJIT> >(m, "PatchEnergyJIT", pybind11::base< hpmc::PatchEnergy >())
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string&,
                                 Scalar >())
            .def("getRCut", &PatchEnergyJIT::getRCut)
            .def("energy", &PatchEnergyJIT::energy);
    }
