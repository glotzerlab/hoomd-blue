#include "StreamingGeometry.h"

namespace mpcd
{
namespace detail
{

//! Export boundary enum to python
void export_boundary(pybind11::module& m)
    {
    namespace py = pybind11;
    py::enum_<mpcd::detail::boundary>(m, "boundary")
        .value("no_slip", mpcd::detail::boundary::no_slip)
        .value("slip", mpcd::detail::boundary::slip);
    }

//! Export BulkGeometry to python
void export_BulkGeometry(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::detail::BulkGeometry, std::shared_ptr<mpcd::detail::BulkGeometry> >(m, "BulkGeometry")
        .def(py::init<>());
    }

//! Export SlitGeometry to python
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
