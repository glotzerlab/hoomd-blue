#include "PythonLocalDataAccess.h"


void export_HOOMDHostBuffer(pybind11::module &m)
    {
    pybind11::class_<HOOMDHostBuffer>(
        m, "HOOMDHostBuffer", pybind11::buffer_protocol())
        .def_buffer([](HOOMDHostBuffer &b) -> pybind11::buffer_info 
                {
                return b.new_buffer();
                })
        .def_property_readonly("shape", &HOOMDHostBuffer::getShape)
        .def_property_readonly("dtype", &HOOMDHostBuffer::getType)
        ;
    }
