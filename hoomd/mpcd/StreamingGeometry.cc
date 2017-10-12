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
    py::class_<BulkGeometry, std::shared_ptr<BulkGeometry> >(m, "BulkGeometry")
        .def(py::init<>());
    }

void export_SlitGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<SlitGeometry, std::shared_ptr<SlitGeometry> >(m, "SlitGeometry")
        .def(py::init<Scalar, Scalar, boundary>())
        .def_property("H", &SlitGeometry::getH, &SlitGeometry::setH)
        .def_property("V", &SlitGeometry::getVelocity, &SlitGeometry::setVelocity)
        .def_property("boundary", &SlitGeometry::getBoundaryCondition, &SlitGeometry::setBoundaryCondition);
    }

} // end namespace detail
} // end namespace mpcd
