#include "c_tensor_network.h"
#include <pybind11/stl.h>

TensorNetwork::TensorNetwork() = default;
size_t TensorNetwork::n_tensors() const { return nodes_.size(); }

void TensorNetwork::add_tensor(const std::string &id, const Tensor &tensor) {
    auto buf_info = tensor.request();
    std::vector<ssize_t> shape(buf_info.shape.begin(), buf_info.shape.end());
    NodeData nd{tensor, shape, std::vector<bool>(shape.size(), true)};
    nodes_map_[id] = nodes_.size();
    nodes_.push_back(std::move(nd));
}

void TensorNetwork::add_measurement(const std::string &id, double alpha, double beta) {
    Tensor t({2});
    auto buf = t.mutable_data();
    buf[0] = Complex(alpha, 0);
    buf[1] = Complex(beta, 0);
    add_tensor(id, t);
}

void TensorNetwork::add_pauli_pair(const std::string &id, const std::string &p0, const std::string &p1) {
    static auto paulis = [&]() {
        std::unordered_map<std::string, Tensor> m;
        py::module utils = py::module::import("tncdr.evolutors.tensor_network.utils");
        auto dict = utils.attr("paulis");
        for (auto &k : dict.attr("keys")().cast<std::vector<std::string>>()) {
            m[k] = dict.attr(k.c_str()).cast<Tensor>();
        }
        return m;
    }();
    const auto &a = paulis.at(p0);
    const auto &b = paulis.at(p1);
    auto bufA = a.request(); std::vector<ssize_t> shape(bufA.shape.begin(), bufA.shape.end());
    shape[0] *= 2;
    Tensor stacked(shape);
    auto sa = a.unchecked<2>(); auto sb = b.unchecked<2>(); auto su = stacked.mutable_unchecked<2>();
    for (ssize_t i = 0; i < shape[1]; ++i) {
        su(0, i) = sa(0, i);
        su(1, i) = sb(0, i);
    }
    add_tensor(id, stacked);
}

void TensorNetwork::add_copy_tensor(const std::string &id, int n) {
    Tensor t({n, n, n});
    auto buf = t.mutable_data();
    size_t N = static_cast<size_t>(n) * n * n;
    std::fill(buf, buf + N, Complex(0, 0));
    for (int i = 0; i < n; ++i) buf[i * n * n + i * n + i] = Complex(1, 0);
    add_tensor(id, t);
}

void TensorNetwork::add_edge(const std::string &in, const std::string &out, const std::string &eid, std::pair<int,int> dirs) {
    int i = idx(in), o = idx(out);
    auto &ni = nodes_[i], &no = nodes_[o];
    if (ni.shape[dirs.first] != no.shape[dirs.second])
        throw std::runtime_error("Incompatible dims");
    if (!ni.free_legs[dirs.first] || !no.free_legs[dirs.second])
        throw std::runtime_error("Leg already used");
    ni.free_legs[dirs.first] = false;
    no.free_legs[dirs.second] = false;
    edges_.push_back({i, o, EdgeData{eid, dirs}});
}

void TensorNetwork::remove_edge(const std::string &in, const std::string &out, const std::string &eid) {
    int i = idx(in), o = idx(out);
    auto it = std::find_if(edges_.begin(), edges_.end(), [&](auto &e){ return e.i==i && e.o==o && e.data.id==eid; });
    if (it == edges_.end()) throw std::runtime_error("Edge not found");
    nodes_[i].free_legs[it->data.dirs.first] = true;
    nodes_[o].free_legs[it->data.dirs.second] = true;
    edges_.erase(it);
}

void TensorNetwork::complex_conjugate() {
    for (auto &nd : nodes_) {
        auto buf = nd.tensor.mutable_data();
        for (size_t i = 0; i < nd.tensor.size(); ++i)
            buf[i] = std::conj(buf[i]);
    }
    std::unordered_map<std::string,int> nm;
    for (auto &kv : nodes_map_) nm[kv.first + "_dg"] = kv.second;
    nodes_map_.swap(nm);
}

py::object TensorNetwork::draw(bool show_labels, const std::string &title) {
    (void)show_labels; (void)title;
    return py::none();
}

void TensorNetwork::contract(const std::string &in, const std::string &out, const std::vector<std::string> &eids, const std::string &new_id) {
    int i = idx(in), o = idx(out);
    auto [axes_i, axes_o] = collect_axes(i, o, eids);
    const auto &A = nodes_[i], &B = nodes_[o];
    auto compA = complement(A.shape.size(), axes_i);
    auto compB = complement(B.shape.size(), axes_o);

    auto mA = make_mat(A.tensor, A.shape, axes_i, compA, false);
    auto mB = make_mat(B.tensor, B.shape, axes_o, compB, true);
    Eigen::MatrixXcd mC = mA * mB;

    std::vector<ssize_t> new_shape;
    for (auto ax : compA) new_shape.push_back(A.shape[ax]);
    for (auto ax : compB) new_shape.push_back(B.shape[ax]);

    Tensor tout(new_shape);
    unflatten_to_tensor(mC, tout, new_shape);

    remove_node(in);
    remove_node(out);
    add_tensor(new_id, tout);
}

void TensorNetwork::svd_decomposition(const std::string &node, const std::string &left_id, const std::vector<std::string> &le,
                                      const std::string &right_id, const std::vector<std::string> &re,
                                      const std::string &mid, const std::string &ml,
                                      const std::string &mr, int maxb) {
    int nidx = idx(node);
    auto shape = nodes_[nidx].shape;
    auto [axesL, axesR] = collect_axes(nidx, le, re);
    auto compL = complement(shape.size(), axesL);
    auto compR = complement(shape.size(), axesR);

    Eigen::MatrixXcd M = make_mat(nodes_[nidx].tensor, shape, axesL, compR);
    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto S = svd.singularValues();
    int chi = S.size(); if (maxb>0) chi = std::min(chi, maxb);

    Eigen::MatrixXcd U = svd.matrixU().leftCols(chi);
    Eigen::MatrixXcd V = svd.matrixV().leftCols(chi).adjoint();
    Eigen::VectorXcd Sv = S.head(chi);

    // Build tensor U
    std::vector<ssize_t> shapeU;
    for (auto ax : compL) shapeU.push_back(shape[ax]);
    shapeU.push_back(chi);
    Tensor TU(shapeU);
    unflatten_to_tensor(U, TU, shapeU);

    // Build tensor V^
    std::vector<ssize_t> shapeV;
    shapeV.push_back(chi);
    for (auto ax : compR) shapeV.push_back(shape[ax]);
    Tensor TV(shapeV);
    unflatten_to_tensor(V, TV, shapeV);

    // Build Lambda
    Tensor TL({chi, chi});
    auto bufL = TL.mutable_data();
    for (int i = 0; i < chi*chi; ++i) bufL[i] = Complex(0,0);
    for (int i = 0; i < chi; ++i) bufL[i*chi + i] = Sv[i];

    remove_node(node);
    add_tensor(left_id, TU);
    add_tensor(right_id, TV);
    add_tensor(mid, TL);
    add_edge(left_id, mid, ml, {(int)axesL.size(), 0});
    add_edge(right_id, mid, mr, {0, 1});
}

TensorNetwork TensorNetwork::merge_tns(const TensorNetwork &t1, const TensorNetwork &t2) {
    TensorNetwork out = t1;
    int offset = static_cast<int>(out.nodes_.size());
    for (auto &kv : t2.nodes_map_) {
        const auto &nd = t2.nodes_[kv.second];
        out.nodes_map_[kv.first] = offset + kv.second;
        out.nodes_.push_back(nd);
    }
    for (auto &er : t2.edges_) {
        out.edges_.push_back({er.i + offset, er.o + offset, er.data});
    }
    return out;
}

int TensorNetwork::idx(const std::string &name) const {
    auto it = nodes_map_.find(name);
    if (it == nodes_map_.end()) throw std::runtime_error("Node not found");
    return it->second;
}

void TensorNetwork::remove_node(const std::string &name) {
    int i = idx(name);
    nodes_.erase(nodes_.begin() + i);
    nodes_map_.erase(name);
    // Note: edges and other indices may need adjustment
}

std::vector<int> TensorNetwork::complement(int n, const std::vector<int> &axes) {
    std::vector<bool> mark(n,false);
    for (int a : axes) mark[a] = true;
    std::vector<int> out;
    for (int i = 0; i < n; ++i) if (!mark[i]) out.push_back(i);
    return out;
}

std::pair<std::vector<int>,std::vector<int>> TensorNetwork::collect_axes(int i, int o, const std::vector<std::string> &eids) const {
    std::vector<int> ai, ao;
    for (auto &eid : eids) for (auto &er : edges_) if (er.i==i && er.o==o && er.data.id==eid) {
        ai.push_back(er.data.dirs.first);
        ao.push_back(er.data.dirs.second);
    }
    return {ai, ao};
}

std::pair<std::vector<int>,std::vector<int>> TensorNetwork::collect_axes(int i, const std::vector<std::string> &le, const std::vector<std::string> &re) const {
    std::vector<int> li, ri;
    for (auto &er : edges_) for (auto &eid : le) if (er.i==i && er.data.id==eid) li.push_back(er.data.dirs.first);
    for (auto &er : edges_) for (auto &eid : re) if (er.o==i && er.data.id==eid) ri.push_back(er.data.dirs.second);
    return {li, ri};
}

Eigen::MatrixXcd TensorNetwork::make_mat(const Tensor &T, const std::vector<ssize_t> &shape,
                                         const std::vector<int> &axesC, const std::vector<int> &axesNC,
                                         bool transpose) {
    auto buf = T.unchecked<1>(); // raw pointer access will use request if needed
    auto strides = compute_strides(shape);
    int dimC = 1, dimNC = 1;
    for (int a : axesC) dimC *= shape[a];
    for (int a : axesNC) dimNC *= shape[a];
    Eigen::MatrixXcd M(transpose?dimC:dimNC, transpose?dimNC:dimC);
    std::vector<ssize_t> idx(shape.size());
    for (int nc = 0; nc < dimNC; ++nc) {
        auto cNC = unravel(nc, axesNC, shape, strides);
        for (int c = 0; c < dimC; ++c) {
            auto cC = unravel(c, axesC, shape, strides);
            for (size_t k = 0; k < axesNC.size(); ++k) idx[axesNC[k]] = cNC[k];
            for (size_t k = 0; k < axesC.size(); ++k) idx[axesC[k]] = cC[k];
            Complex v = T.data()[dot(idx, strides)];
            if (!transpose) M(nc, c) = v; else M(c, nc) = v;
        }
    }
    return M;
}

void TensorNetwork::unflatten_to_tensor(const Eigen::MatrixXcd &M, Tensor &T, const std::vector<ssize_t> &shape) {
    auto strides = compute_strides(shape);
    int rows = M.rows(), cols = M.cols();
    std::vector<ssize_t> idx(shape.size());
    for (int r = 0; r < rows; ++r) {
        auto cNC = unravel(r, shape);
        for (int c = 0; c < cols; ++c) {
            auto cC = unravel(c, shape);
            for (size_t k = 0; k < cNC.size(); ++k) idx[k] = cNC[k];
            for (size_t k = 0; k < cC.size(); ++k) idx[k + cNC.size()] = cC[k];
            T.mutable_data()[dot(idx, strides)] = M(r, c);
        }
    }
}

std::vector<ssize_t> TensorNetwork::compute_strides(const std::vector<ssize_t> &shape) {
    int n = shape.size();
    std::vector<ssize_t> s(n);
    s[n-1] = 1;
    for (int i = n-2; i >= 0; --i) s[i] = s[i+1] * shape[i+1];
    return s;
}

ssize_t TensorNetwork::dot(const std::vector<ssize_t> &a, const std::vector<ssize_t> &b) {
    ssize_t r = 0;
    for (size_t i = 0; i < a.size(); ++i) r += a[i] * b[i];
    return r;
}

std::vector<ssize_t> TensorNetwork::unravel(int idx, const std::vector<int> &axes,
                                            const std::vector<ssize_t> &shape,
                                            const std::vector<ssize_t> &strides) {
    std::vector<ssize_t> out(axes.size());
    for (size_t i = 0; i < axes.size(); ++i) out[i] = (idx / strides[axes[i]]) % shape[axes[i]];
    return out;
}

std::vector<ssize_t> TensorNetwork::unravel(int idx, const std::vector<ssize_t> &shape) {
    auto s = compute_strides(shape);
    std::vector<ssize_t> out(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) out[i] = (idx / s[i]) % shape[i];
    return out;
}
