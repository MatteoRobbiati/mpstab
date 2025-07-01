#include "c_tensor_network.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(c_tensor_network, m) {
    py::class_<TensorNetwork>(m, "TensorNetwork")
        .def(py::init<>())
        .def_property_readonly("n_tensors", &TensorNetwork::n_tensors)
        .def("add_tensor", &TensorNetwork::add_tensor)
        .def("add_measurement", &TensorNetwork::add_measurement,
             py::arg("id"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0)
        .def("add_pauli_pair", &TensorNetwork::add_pauli_pair)
        .def("add_copy_tensor", &TensorNetwork::add_copy_tensor)
        .def("add_edge", &TensorNetwork::add_edge)
        .def("remove_edge", &TensorNetwork::remove_edge)
        .def("complex_conjugate", &TensorNetwork::complex_conjugate)
        .def("draw", &TensorNetwork::draw)
        .def("contract", &TensorNetwork::contract)
        .def("svd_decomposition", &TensorNetwork::svd_decomposition,
             py::arg("node"), py::arg("left_id"), py::arg("left_edges"),
             py::arg("right_id"), py::arg("right_edges"),
             py::arg("middle_id") = "Lambda", py::arg("medge_l") = "chi",
             py::arg("medge_r") = "chi", py::arg("max_bond") = -1)
        .def_static("merge_tns", &TensorNetwork::merge_tns);
}#include "c_tensor_network.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(c_tensor_network, m) {
    py::class_<TensorNetwork>(m, "TensorNetwork")
        .def(py::init<>())
        .def_property_readonly("n_tensors", &TensorNetwork::n_tensors)
        .def("add_tensor", &TensorNetwork::add_tensor)
        .def("add_measurement", &TensorNetwork::add_measurement,
             py::arg("id"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0)
        .def("add_pauli_pair", &TensorNetwork::add_pauli_pair)
        .def("add_copy_tensor", &TensorNetwork::add_copy_tensor)
        .def("add_edge", &TensorNetwork::add_edge)
        .def("remove_edge", &TensorNetwork::remove_edge)
        .def("complex_conjugate", &TensorNetwork::complex_conjugate)
        .def("draw", &TensorNetwork::draw)
        .def("contract", &TensorNetwork::contract)
        .def("svd_decomposition", &TensorNetwork::svd_decomposition,
             py::arg("node"), py::arg("left_id"), py::arg("left_edges"),
             py::arg("right_id"), py::arg("right_edges"),
             py::arg("middle_id") = "Lambda", py::arg("medge_l") = "chi",
             py::arg("medge_r") = "chi", py::arg("max_bond") = -1)
        .def_static("merge_tns", &TensorNetwork::merge_tns);
}
