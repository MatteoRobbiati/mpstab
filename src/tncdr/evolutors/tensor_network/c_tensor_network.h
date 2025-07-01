#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace py = pybind11;
using Complex = std::complex<double>;
using Tensor = py::array_t<Complex>;

struct NodeData {
    Tensor tensor;
    std::vector<ssize_t> shape;
    std::vector<bool> free_legs;
};

struct EdgeData {
    std::string id;
    std::pair<int,int> dirs;
};

class TensorNetwork {
public:
    TensorNetwork();
    size_t n_tensors() const;

    void add_tensor(const std::string &id, const Tensor &tensor);
    void add_measurement(const std::string &id, double alpha=1.0, double beta=0.0);
    void add_pauli_pair(const std::string &id, const std::string &p0, const std::string &p1);
    void add_copy_tensor(const std::string &id, int n);

    void add_edge(const std::string &in, const std::string &out,
                  const std::string &eid, std::pair<int,int> dirs);
    void remove_edge(const std::string &in, const std::string &out,
                     const std::string &eid);

    void complex_conjugate();
    py::object draw(bool show_labels=false, const std::string &title="");

    void contract(const std::string &in, const std::string &out,
                  const std::vector<std::string> &eids,
                  const std::string &new_id);

    void svd_decomposition(const std::string &node,
                           const std::string &left_id,
                           const std::vector<std::string> &left_edges,
                           const std::string &right_id,
                           const std::vector<std::string> &right_edges,
                           const std::string &middle_id = "Lambda",
                           const std::string &medge_l = "chi",
                           const std::string &medge_r = "chi",
                           int max_bond = -1);

    static TensorNetwork merge_tns(const TensorNetwork &tn1,
                                   const TensorNetwork &tn2);

private:
    struct EdgeRec { int i, o; EdgeData data; };
    std::vector<NodeData> nodes_;
    std::vector<EdgeRec> edges_;
    std::unordered_map<std::string,int> nodes_map_;

    int idx(const std::string &name) const;
    void remove_node(const std::string &name);

    static std::vector<int> complement(int n, const std::vector<int> &axes);
    std::pair<std::vector<int>,std::vector<int>> collect_axes(int i, int o,
        const std::vector<std::string> &eids) const;
    std::pair<std::vector<int>,std::vector<int>> collect_axes(int i,
        const std::vector<std::string> &left,
        const std::vector<std::string> &right) const;

    static Eigen::MatrixXcd make_mat(const Tensor &T,
                                     const std::vector<ssize_t> &shape,
                                     const std::vector<int> &axesC,
                                     const std::vector<int> &axesNC,
                                     bool transpose=false);
    static void unflatten_to_tensor(const Eigen::MatrixXcd &M,
                                    Tensor &T,
                                    const std::vector<ssize_t> &shape);
    static std::vector<ssize_t> compute_strides(const std::vector<ssize_t> &shape);
    static ssize_t dot(const std::vector<ssize_t> &a,
                       const std::vector<ssize_t> &b);
    static std::vector<ssize_t> unravel(int idx,
                                        const std::vector<int> &axes,
                                        const std::vector<ssize_t> &shape,
                                        const std::vector<ssize_t> &strides);
    static std::vector<ssize_t> unravel(int idx,
                                        const std::vector<ssize_t> &shape);
};
