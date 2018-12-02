#include "ForceEnergyJIT.h"
#include "ForceEvalFactory.h"

// Set preprocessor variable to avoid compiling cereal files that throw exceptions when using LLVM (which sets -fno-exceptions.
#define NO_CEREAL_INCLUDE
#include "hoomd/BoxDim.h"
#include <sstream>

#define FORCE_ENERGY_LOG_NAME           "force_energy"
/*! \param exec_conf The execution configuration (used for messages and MPI communication)
    \param llvm_ir Contents of the LLVM IR to load

    After construction, the LLVM IR is loaded, compiled, and the energy() method is ready to be called.
*/
ForceEnergyJIT::ForceEnergyJIT(std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir)
    {
    // build the JIT.
    m_factory = std::shared_ptr<ForceEvalFactory>(new ForceEvalFactory(llvm_ir));

    // get the evaluator
    m_eval = m_factory->getEval();

    if (!m_eval)
        {
        exec_conf->msg->error() << m_factory->getError() << std::endl;
        throw std::runtime_error("Error compiling JIT code.");
        }
    }


void export_ForceEnergyJIT(pybind11::module &m)
    {
      pybind11::class_<hpmc::ForceEnergy, std::shared_ptr<hpmc::ForceEnergy> >(m, "ForceEnergy")
              .def(pybind11::init< >());
    pybind11::class_<ForceEnergyJIT, std::shared_ptr<ForceEnergyJIT> >(m, "ForceEnergyJIT", pybind11::base< hpmc::ForceEnergy >())
            .def(pybind11::init< std::shared_ptr<ExecutionConfiguration>,
                                 const std::string& >())
            .def("energy", &ForceEnergyJIT::energy);
    }
