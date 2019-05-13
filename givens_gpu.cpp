#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <pybind11/eigen.h>
#include <stdio.h>
#include <assert.h>
#include <string>
#include "util.h"

namespace py = pybind11;

float* run_minimize_subspaces_l1(float *d_U, int d);
std::tuple<float,int,int>* optimize(float *h_U, int d, int max_iter, std::string method);
float* compute_cost_matrix(int, float*, float*);


Eigen::VectorXf minimize_all_subspaces(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U, std::string method){
    int d = U.cols();
    int n = d * (d-1) / 2;
    cudaError_t error;

    int data_size = d*d;
    float *gpu_ptr;
    cudaMalloc(&gpu_ptr, data_size * sizeof(float));
    CUDA_CHECK;

    cudaMemcpy(gpu_ptr, U.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);

    float *h_metrics = new float[n];
    if(method == "l1"){
        h_metrics = run_minimize_subspaces_l1(gpu_ptr, d);
    }else{
        throw std::runtime_error("Method not known!");
    }

    cudaFree(gpu_ptr);
    CUDA_CHECK;

    Eigen::Map<Eigen::VectorXf> metrics_vec(h_metrics, n);
    return metrics_vec;
}


Eigen::VectorXf minimize_all_subspaces_l1(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U){
    return minimize_all_subspaces(U, "l1");
}


std::vector<std::tuple<float,int,int>> launch_optimizer(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U, int max_iter, std::string method){
    std::tuple<float,int,int> *trajectory_ptr;
    trajectory_ptr = optimize(U.data(), U.rows(), max_iter, method);
    std::vector<std::tuple<float,int,int>> trajectory;
    trajectory.reserve(max_iter);
    for(int n=0; n < max_iter; n++){
        trajectory.push_back(trajectory_ptr[n]);
    }
    return trajectory;
}


std::vector<std::tuple<float,int,int>> coordinate_descent_l1(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> U, int max_iter){
    return launch_optimizer(U, max_iter, "l1");
}


Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> build_cost_matrix(Eigen::MatrixXf U, Eigen::MatrixXf V){
    int d = U.cols();
    float *h_C = new float[2*d*d]();
    h_C = compute_cost_matrix(d, U.data(), V.data());
    return Eigen::Map<Eigen::MatrixXf>(h_C, d, 2*d);
}


PYBIND11_MODULE(givens_gpu, m)
{
    m.def("coordinate_descent_l1", coordinate_descent_l1);
    m.def("minimize_all_subspaces_l1", minimize_all_subspaces_l1);
    m.def("build_cost_matrix", build_cost_matrix);
}
