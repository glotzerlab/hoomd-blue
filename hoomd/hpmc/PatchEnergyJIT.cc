#include "PatchEnergyJIT.h"
#include "EvalFactory.h"

#include <sstream>

/*! \param exec_conf The execution configuration (used for messages and MPI communication)
    \param llvm_ir Contents of the LLVM IR to load
    \param r_cut Center to center distance beyond which the patch energy is 0

    After construction, the LLVM IR is loaded, compiled, and the energy() method is ready to be
   called.
*/
PatchEnergyJIT::PatchEnergyJIT(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ExecutionConfiguration> exec_conf,
                               const std::string& cpu_code,
                               const std::vector<std::string>& compiler_args,
                               Scalar r_cut,
                               pybind11::array_t<float> param_array)
    : PatchEnergy(sysdef), m_exec_conf(exec_conf), m_r_cut(r_cut),
      m_alpha_size(static_cast<unsigned int>(param_array.size())),
      m_alpha(param_array.data(),
              param_array.data() + param_array.size(),
              managed_allocator<float>(m_exec_conf->isCUDAEnabled()))
    {
    // build the JIT.
    m_factory = std::shared_ptr<EvalFactory>(new EvalFactory(cpu_code, compiler_args));

    // save the C++ code string for exporting to python
    m_cpu_code = cpu_code;

    // get the evaluator
    m_eval = m_factory->getEval();

    if (!m_eval)
        {
        std::ostringstream s;
        s << "Error compiling JIT code:" << std::endl;
        s << cpu_code << std::endl;
        s << m_factory->getError() << std::endl;
        throw std::runtime_error(s.str());
        }

    m_factory->setAlphaArray(&m_alpha.front());
    }

void export_PatchEnergyJIT(pybind11::module& m)
    {
    pybind11::class_<hpmc::PatchEnergy, std::shared_ptr<hpmc::PatchEnergy>>(m, "PatchEnergy")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    pybind11::class_<PatchEnergyJIT, Compute, hpmc::PatchEnergy, std::shared_ptr<PatchEnergyJIT>>(
        m,
        "PatchEnergyJIT")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExecutionConfiguration>,
                            const std::string&,
                            const std::vector<std::string>&,
                            Scalar,
                            pybind11::array_t<float>>())
        .def_property("r_cut", &PatchEnergyJIT::getRCut, &PatchEnergyJIT::setRCut)
        .def_property_readonly("array_size", &PatchEnergyJIT::getArraySize)
        .def("energy", &PatchEnergyJIT::energy)
        .def_property_readonly("param_array", &PatchEnergyJIT::getParamArray)
#ifdef ENABLE_MPI
        .def("setCommunicator", &Compute::setCommunicator)
        .def_property_readonly("llvm_ir", &PatchEnergyJIT::llvm_ir)
#endif
        ;
    }
