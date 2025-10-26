#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "iam/core/Tensor.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_iam_core, m) {
    py::class_<iam::Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def("shape", &iam::Tensor::shape)
        .def("size", &iam::Tensor::size);
}
