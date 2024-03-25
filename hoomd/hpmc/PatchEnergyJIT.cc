// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PatchEnergyJIT.h"
#include "EvalFactory.h"

#include <sstream>

namespace hoomd
    {
namespace hpmc
    {
/*! \param exec_conf The execution configuration (used for messages and MPI communication).
    \param cpu_code C++ code to compile.
    \param compiler_args Additional arguments to pass to the compiler.
    \param r_cut Center to center distance beyond which the patch energy is 0.
    \param param_array Values for the parameter array.

    After construction, the C++ code is compiled, and the energy() method is ready to be
   called.
*/
PatchEnergyJIT::PatchEnergyJIT(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ExecutionConfiguration> exec_conf,
                               const std::string& cpu_code,
                               const std::vector<std::string>& compiler_args,
                               Scalar r_cut,
                               pybind11::array_t<float> param_array,
                               bool is_union)
    : PatchEnergy(sysdef), m_exec_conf(exec_conf), m_r_cut_isotropic(r_cut),
      m_param_array(param_array.data(),
                    param_array.data() + param_array.size(),
                    hoomd::detail::managed_allocator<float>(m_exec_conf->isCUDAEnabled())),
      m_is_union(is_union)
    {
    // build the JIT.
    EvalFactory* factory = new EvalFactory(cpu_code, compiler_args, this->m_is_union);

    // get the evaluator
    m_eval = factory->getEval();

    if (!m_eval)
        {
        std::ostringstream s;
        s << "Error compiling JIT code:" << std::endl;
        s << cpu_code << std::endl;
        s << factory->getError() << std::endl;
        throw std::runtime_error(s.str());
        }

    factory->setAlphaArray(&m_param_array.front());
    m_factory = std::shared_ptr<EvalFactory>(factory);
    }

namespace detail
    {
void export_PatchEnergyJIT(pybind11::module& m)
    {
    pybind11::class_<PatchEnergyJIT, hpmc::PatchEnergy, std::shared_ptr<PatchEnergyJIT>>(
        m,
        "PatchEnergyJIT")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ExecutionConfiguration>,
                            const std::string&,
                            const std::vector<std::string>&,
                            Scalar,
                            pybind11::array_t<float>>())
        .def_property("r_cut", &PatchEnergyJIT::getRCut, &PatchEnergyJIT::setRCut)
        .def("energy", &PatchEnergyJIT::energy)
        .def_property_readonly("param_array", &PatchEnergyJIT::getParamArray);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
