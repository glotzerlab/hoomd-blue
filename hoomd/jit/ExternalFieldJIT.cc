#include "ExternalFieldJIT.h"
#include "ExternalFieldEvalFactory.h"

#include <sstream>

#define EXTERNAL_FIELD_JIT_LOG_NAME           "jit_energy"
/*! \param exec_conf The execution configuration (used for messages and MPI communication)
    \param llvm_ir Contents of the LLVM IR to load

    After construction, the LLVM IR is loaded, compiled, and the energy() method is ready to be called.
*/

template< class Shape >
ExternalFieldJIT<Shape>::ExternalFieldJIT(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ExecutionConfiguration> exec_conf, const std::string& llvm_ir) : hpmc::ExternalFieldMono<Shape>(sysdef)
    {
    // build the JIT.
    m_factory = std::shared_ptr<ExternalFieldEvalFactory>(new ExternalFieldEvalFactory(llvm_ir));

    // get the evaluator
    m_eval = m_factory->getEval();

    if (!m_eval)
        {
        exec_conf->msg->error() << m_factory->getError() << std::endl;
        throw std::runtime_error("Error compiling JIT code.");
        }
    }


template< class Shape>
void export_ExternalFieldJIT(pybind11::module &m, std::string name)
    {
    pybind11::class_<ExternalFieldJIT<Shape>, std::shared_ptr<ExternalFieldJIT<Shape> > >(m, name.c_str(), pybind11::base< hpmc::ExternalFieldMono <Shape> >())
            .def(pybind11::init< std::shared_ptr<SystemDefinition>, 
                                 std::shared_ptr<ExecutionConfiguration>,
                                 const std::string& >())
            .def("energy", &ExternalFieldJIT<Shape>::energy);
    }
