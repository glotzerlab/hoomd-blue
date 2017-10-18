#include "StreamingGeometry.h"

namespace mpcd
{
namespace detail
{

void export_boundary(pybind11::module& m)
    {
    namespace py = pybind11;
    py::enum_<boundary>(m, "boundary")
        .value("no_slip", boundary::no_slip)
        .value("slip", boundary::slip);
    }

void export_BulkGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<BulkGeometry, std::shared_ptr<const BulkGeometry> >(m, "BulkGeometry")
        .def(py::init<>());
    }

void export_SlitGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SlitGeometry, std::shared_ptr<const SlitGeometry> >(m, "SlitGeometry")
        .def(py::init<Scalar, Scalar, boundary>())
        .def("getH", &SlitGeometry::getH)
        .def("getVelocity", &SlitGeometry::getVelocity)
        .def("getBoundaryCondition", &SlitGeometry::getBoundaryCondition);
    }

} // end namespace detail
} // end namespace mpcd
