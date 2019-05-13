#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cub/cub.cuh>
#include <assert.h>
#include "util.h"

#define PI 3.141592654f

struct CoordinateDescentData {
    /*
       Carries data and temporary storage throught the algorithm iterations.
    */
    int d;
    float *d_U;
    float *d_metrics;
    float *d_metrics_selected;
    int   *d_max_cols;
    float *d_angles;
    int   h_opt_subspace[2];
    float h_opt_angle[3];
    int   num_metric_elements;
    int   num_selected_elements;
    int   num_selected_and_col_elements;
    int   num_segments;
    void  *d_temp_storage_col_max;
    size_t temp_storage_bytes_col_max;
    void  *d_temp_storage_metric_max;
    size_t temp_storage_bytes_metric_max;
    int   *d_offsets;
    cub::KeyValuePair<int, float>   *d_row_argmax;
    cub::KeyValuePair<int, float>   *d_metric_argmax;

    CoordinateDescentData(float *h_U, int d_){
        d = d_;
        h_opt_subspace[0] = -1;
        h_opt_subspace[1] = -1;
        h_opt_angle[0] = 0.f;
        h_opt_angle[1] = 0.f;
        h_opt_angle[2] = 0.f;
        num_metric_elements = d*(d-1)/2;
        num_selected_elements = 2*(d-1);
        num_selected_and_col_elements = d*num_selected_elements;
        num_segments = num_selected_elements;
        cudaMalloc(&d_U, (d*d) * sizeof(float));
        CUDA_CHECK;
        cudaMemcpy(d_U, h_U, (d*d) * sizeof(float), cudaMemcpyHostToDevice);
        CUDA_CHECK;
        cudaMalloc(&d_metrics, num_metric_elements * sizeof(float));
        CUDA_CHECK;
        cudaMalloc(&d_metrics_selected, num_selected_and_col_elements * sizeof(float));
        CUDA_CHECK;
        cudaMalloc(&d_max_cols, num_metric_elements * sizeof(int));
        CUDA_CHECK;
        cudaMalloc(&d_angles, num_metric_elements * sizeof(float));
        CUDA_CHECK;
        cudaMalloc(&d_offsets, (num_segments+1) * sizeof(int));
        CUDA_CHECK;
        cudaMalloc(&d_row_argmax, num_segments * sizeof(cub::KeyValuePair<int, float>));
        CUDA_CHECK;

        //temp data for column reduction
        int *h_offsets = new int[num_segments+1];
        for(int k=0; k < num_segments+1; k++){
            h_offsets[k] = k * d;
        }
        cudaMemcpy(d_offsets, h_offsets, (num_segments+1) * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_CHECK;
        d_temp_storage_col_max = NULL;
        temp_storage_bytes_col_max = 0;
        cub::DeviceSegmentedReduce::ArgMax(d_temp_storage_col_max, temp_storage_bytes_col_max, d_metrics_selected, d_row_argmax, num_segments, d_offsets, d_offsets+1);
        cudaMalloc(&d_temp_storage_col_max, temp_storage_bytes_col_max);
        CUDA_CHECK;

        //temp data for metric reduction
        cudaMalloc(&d_metric_argmax, sizeof(cub::KeyValuePair<int, float>));
        CUDA_CHECK;
        d_temp_storage_metric_max = NULL;
        temp_storage_bytes_metric_max = 0;
        cub::DeviceReduce::ArgMax(d_temp_storage_metric_max, temp_storage_bytes_metric_max, d_metrics, d_metric_argmax, num_metric_elements);
        cudaMalloc(&d_temp_storage_metric_max, temp_storage_bytes_metric_max);
        CUDA_CHECK;
    };

    void FreeAll(){
        cudaFree(d_U);
        cudaFree(d_metrics);
        cudaFree(d_metrics_selected);
        cudaFree(d_max_cols);
        cudaFree(d_temp_storage_col_max);
        cudaFree(d_temp_storage_metric_max);
        cudaFree(d_offsets);
        cudaFree(d_row_argmax);
    }

};


__host__ __device__ void to_tril_idx(int d, int lin_idx, int *tril_idx){
    /*
        Converts a linear index to triangular inddices, i.e., a linear index up to length
        d(d-1)/2 is mapped to (i,j) coordinates of an upper triangular matrix in d dimensions.
    */
    double temp = -8*lin_idx + 4*d*(d-1)-7;
    int i = d - 2 - floor(sqrt(temp)/2.0 - 0.5); 
    int j = lin_idx + i + 1 - d*(d-1)/2 + (d-i)*((d-i)-1)/2;
    tril_idx[0] = i;
    tril_idx[1] = j;
}


__host__ __device__ int to_lin_idx(int d, int i, int j){
    /*
        Converts triangular indices to linear index, i.e., (i,j) coordinates of an upper triangular 
        matrix in d dimensions are mapped to a  linear index up to length d(d-1)/2.
    */
    int n = d*(d-1)/2;
    return (n - (d-i) * (d-i - 1)/2 + j - i - 1);
}


__host__ __device__ void selected_lin_idx_to_tril_col(int selected_lin_idx, int d, int i_prev, int j_prev, int *tril_col_idx){
    /*
        Two consecutive linear indices up to 2d(d-1) in d dimensions are mapped to upper triangular plus column index.   
    */
    int n = d*(d-1);
    int m = selected_lin_idx / n;
    int idx = (1-m) * i_prev + m * j_prev;
    int idx_counter = selected_lin_idx - m * n;

    int k = idx_counter % d;
    int r = idx_counter / d;
    int s = (idx + 1 + r) % d;
    int p = (idx + 1 + r) / d;
    int i = (1-p) * idx + p * s;
    int j = p * idx + (1-p) * s;

    tril_col_idx[0] = i;
    tril_col_idx[1] = j;
    tril_col_idx[2] = k;
}


__host__ __device__ void selected_lin_idx_to_tril(int selected_lin_idx, int d, int i_prev, int j_prev, int *tril_idx){
    int n = (d-1);
    int m = selected_lin_idx / n;
    int idx = (1-m) * i_prev + m * j_prev;
    int r = selected_lin_idx - m * n;

    int s = (idx + 1 + r) % d;
    int p = (idx + 1 + r) / d;
    int i = (1-p) * idx + p * s;
    int j = p * idx + (1-p) * s;

    tril_idx[0] = i;
    tril_idx[1] = j;
}

__host__ __device__ float compute_rotation_angle(float x, float y){
    /*
        Computes rotation angle to rotate a point (x,y) onto a coordinate axis.
    */
    float pi_half = PI / 2;
    float a = atan2f(y,x);
    int   q = floorf(fmodf(a/pi_half, 4));
    float q_angle = fmodf(a - q*pi_half, (2*PI));
    float alpha;
    if(q_angle < PI /4){
        alpha = -q_angle;
    } else {
        alpha = PI/2 - q_angle;
    }
    return alpha;
}


struct abs_val{
    /*
        Absolute value functor to use a template parameter.
    */
    abs_val()=default;                                                           
    __device__ float operator()(const float x) const{
        return fabsf(x);
    }                                                                     
}; 


template <typename F>
__device__ void compute_subspace_result(float *U, int d, int i, int j, F const& f, float *max_result_ptr, int col_idx){
    /*
        Computes change in metric specified by template parameter for a subspace (i,j).
    */  
    float x = U[d*i+col_idx];
    float y = U[d*j+col_idx];
    float alpha = compute_rotation_angle(x, y);
    float c = cosf(alpha);
    float s = sinf(alpha);

    float result_for_col = 0.f;
    for(int k=0; k < d; k++){
        x = U[d*i+k];
        y = U[d*j+k];
        result_for_col += f(x) + f(y) - f(c*x - s*y) - f(s*x + c*y);
    }

    max_result_ptr[0] = result_for_col;
}


template <typename F>
__global__ void minimize_selected_subspaces(float *U, int d, int i_prev, int j_prev, float *result){
    /*
        Finds rotation that minimizes all subspaces involing i_prev or j_prev.
    */
    int selected_lin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d*(d-1);
    if(selected_lin_idx >= 2*n){
        return;
    }

    int tril_col_idx[3];
    selected_lin_idx_to_tril_col(selected_lin_idx, d, i_prev, j_prev, tril_col_idx);

    compute_subspace_result<F>(U, d, tril_col_idx[0], tril_col_idx[1], F(), &result[selected_lin_idx], tril_col_idx[2]); 
}


template <typename F>
__global__ void minimize_all_subspaces(float *U, int d, float *result, int *d_max_cols, int col_idx){
    /*
       Minimizes all subspaces in parallel for a fixed column.
    */
    int lin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = d*(d-1)/2;
    if(lin_idx >= n){
        return;
    }
    int tril_idx[2];
    to_tril_idx(d, lin_idx, tril_idx);
    int i = tril_idx[0];
    int j = tril_idx[1];

    float prev_val = result[lin_idx];
    float new_val;
    compute_subspace_result(U, d, i, j, F(), &new_val, col_idx); 
    if(new_val > prev_val){
        result[lin_idx] = new_val;
        d_max_cols[lin_idx] = col_idx;
    }
}


template <typename F>
float* run_minimize_subspaces(float *d_U, int d){
    /*
        Wrapper on minimize_all_subspaces.
    */
    int num_elements = d * (d-1) / 2;
    float *h_metrics = new float[num_elements];
    float *max_metrics = new float[num_elements]();
    float *d_metrics;
    cudaMalloc(&d_metrics, num_elements * sizeof(float));
    thrust::device_ptr<float> dev_ptr(d_metrics);
    thrust::fill(dev_ptr, dev_ptr + num_elements, 0.f);

    int *d_max_cols;
    cudaMalloc(&d_max_cols, num_elements * sizeof(int));

    dim3 dimBlock_ms(256, 1, 1);
    dim3 dimGrid_ms(ceil((double)num_elements / dimBlock_ms.x));
    for(int col_idx=0; col_idx < d; col_idx++){
        minimize_all_subspaces<F><<<dimGrid_ms, dimBlock_ms>>>(d_U, d, d_metrics, d_max_cols, col_idx);
        CUDA_CHECK;
        cudaMemcpy(h_metrics, d_metrics, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        CUDA_CHECK;
        for(int k=0; k < num_elements; k++){
            if(h_metrics[k] > max_metrics[k]){
                max_metrics[k] = h_metrics[k];
            }
        }

    }

    cudaFree(d_metrics);
    cudaFree(d_max_cols);

    return max_metrics;
}


float* run_minimize_subspaces_l1(float *d_U, int d){
    return run_minimize_subspaces<abs_val>(d_U, d);
}


void compute_argmax(float *d_in, int num_items, float *max_ptr, int *argmax_ptr){
    /*
        Compute argmax and max of first num_items entries of d_in.
    */
    cub::KeyValuePair<int, float>   *d_argmax;
    cub::KeyValuePair<int, float>   h_argmax[1];

    cudaMalloc(&d_argmax, sizeof(cub::KeyValuePair<int, float>));

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run argmax-reduction
    cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
    cudaMemcpy(h_argmax, d_argmax, sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost);
    cudaFree(d_argmax);
    cudaFree(d_temp_storage);
    argmax_ptr[0] = h_argmax[0].key;
    max_ptr[0] = h_argmax[0].value;
}


void compute_opt_subspace_and_angle(float *d_U, int d, int argmax, int k, int *h_opt_subspace, float *h_opt_angle){
    /*
        CPU method to compute the optimal rotation angle and copy the optimal subspace.
    */
    int tril_idx[2];
    to_tril_idx(d, argmax, tril_idx);
    h_opt_subspace[0] = tril_idx[0];
    h_opt_subspace[1] = tril_idx[1];
    int i = h_opt_subspace[0];
    int j = h_opt_subspace[1];

    float h_x[] = {0.f};
    float h_y[] = {0.f};
    cudaMemcpy(h_x, &d_U[d*i+k], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, &d_U[d*j+k], sizeof(float), cudaMemcpyDeviceToHost);

    float alpha = compute_rotation_angle(h_x[0], h_y[0]);
    h_opt_angle[0] = alpha;
    h_opt_angle[1] = sinf(alpha);
    h_opt_angle[2] = cosf(alpha);
}


__global__ void take_step(float *U, int d, int i, int j, float s, float c){
    /*
       Takes a coordinate descent step by applying the optimal rotation.
     */
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if(k >= d){
        return;
    }

    float x = U[d*i+k];
    float y = U[d*j+k];

    U[d*i+k] = c*x - s*y;
    U[d*j+k] = s*x + c*y;
}


__global__ void update_subspaces_selected(int d, int i_prev, int j_prev, float *d_metrics, int *d_max_cols, cub::KeyValuePair<int, float>   *d_row_argmax){
    /*
        Updates the metric storage with newly computed selected subspaces.
    */
    int selected_lin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(selected_lin_idx >= 2 * (d-1)){
        return;
    }

    int tril_idx[2];  
    selected_lin_idx_to_tril(selected_lin_idx, d, i_prev, j_prev, tril_idx);
    int lin_idx = to_lin_idx(d, tril_idx[0], tril_idx[1]);
    cub::KeyValuePair<int, float> max_for_lin_idx = d_row_argmax[selected_lin_idx];

    d_metrics[lin_idx]  = max_for_lin_idx.value;
    d_max_cols[lin_idx] = max_for_lin_idx.key;
}


template <typename F>
void coordinate_descent_step(int iteration, CoordinateDescentData &data){
    if(iteration == 0){
        thrust::device_ptr<float> dev_ptr(data.d_metrics);
        thrust::fill(dev_ptr, dev_ptr + data.num_metric_elements, 0.f);
        for(int col_idx=0; col_idx < data.d; col_idx++){
            dim3 dimBlock_ms(64, 1, 1);
            dim3 dimGrid_ms(ceil((double)data.num_metric_elements / dimBlock_ms.x));
            minimize_all_subspaces<F><<<dimGrid_ms, dimBlock_ms>>>(
                    data.d_U, 
                    data.d, 
                    data.d_metrics,
                    data.d_max_cols,
                    col_idx);
            CUDA_CHECK;
        }
    }else{
        //minimze subspaces associated with previous update
        dim3 dimBlock_ms(64, 1, 1);
        dim3 dimGrid_ms(ceil((double) (data.num_selected_and_col_elements) / dimBlock_ms.x));
        minimize_selected_subspaces<F><<<dimGrid_ms, dimBlock_ms>>>(
                data.d_U, 
                data.d, 
                data.h_opt_subspace[0], 
                data.h_opt_subspace[1], 
                data.d_metrics_selected);
        CUDA_CHECK;

        //find maximum over columns
        cub::DeviceSegmentedReduce::ArgMax(
                data.d_temp_storage_col_max, 
                data.temp_storage_bytes_col_max, 
                data.d_metrics_selected, 
                data.d_row_argmax, 
                data.num_segments, 
                data.d_offsets, 
                data.d_offsets+1);
        CUDA_CHECK;

        //update subspaces
        dim3 dimBlock_us(64, 1, 1);
        dim3 dimGrid_us(ceil((double) data.num_selected_elements / dimBlock_us.x));
        update_subspaces_selected<<<dimGrid_us, dimBlock_us>>>(
                data.d, 
                data.h_opt_subspace[0], 
                data.h_opt_subspace[1], 
                data.d_metrics, 
                data.d_max_cols, 
                data.d_row_argmax);
        CUDA_CHECK;
    }
    //find maximum over all subspaces metrics
    cub::DeviceReduce::ArgMax(
            data.d_temp_storage_metric_max, 
            data.temp_storage_bytes_metric_max, 
            data.d_metrics, 
            data.d_metric_argmax, 
            data.num_metric_elements);
    CUDA_CHECK;

    cub::KeyValuePair<int, float> h_argmax;
    cudaMemcpy(&h_argmax, data.d_metric_argmax, sizeof(cub::KeyValuePair<int, float>), cudaMemcpyDeviceToHost);
    int max_lin_idx = h_argmax.key;
    int max_col;
    cudaMemcpy(&max_col, &data.d_max_cols[max_lin_idx], sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    compute_opt_subspace_and_angle(
            data.d_U, 
            data.d, 
            max_lin_idx, 
            max_col,
            data.h_opt_subspace, 
            data.h_opt_angle);

    dim3 dimBlock_ts(64,1,1);
    dim3 dimGrid_ts(ceil((double) data.d / dimBlock_ts.x));
    take_step<<<dimGrid_ts, dimBlock_ts>>>(
            data.d_U, 
            data.d, 
            data.h_opt_subspace[0], 
            data.h_opt_subspace[1], 
            data.h_opt_angle[1], 
            data.h_opt_angle[2]);
    CUDA_CHECK;
}


void coordinate_descent_step_l1(int iteration, CoordinateDescentData &data){
    return coordinate_descent_step<abs_val>(iteration, data);
}


std::tuple<float,int,int>* optimize(float *h_U, int d, int max_iter, std::string method){
    std::tuple<float,int,int>* trajectory_ptr = new std::tuple<float,int,int>[max_iter];

    CoordinateDescentData data(h_U, d);

    for(int it=0; it < max_iter; it++){
        if(method == "l1"){
            coordinate_descent_step_l1(it, data);
        }else{
            throw std::runtime_error("Method not known!");
        }
        trajectory_ptr[it] = std::make_tuple(data.h_opt_angle[0], data.h_opt_subspace[0], data.h_opt_subspace[1]);
    }

    data.FreeAll();

    return trajectory_ptr;
}


__global__ void cost_matrix_kernel(int d, float *U, float *V, float* C){
    int i = threadIdx.x + blockDim.x * blockIdx.x;    
    int j_c = threadIdx.y + blockDim.y * blockIdx.y;    
    int j = j_c % d;
    if((i >= d) || (j_c >= 2*d)){
        return;
    }
    int p = j_c/d;
    float v_sign = (float)(1-2*p);
    float sum_of_squares = 0.f;
    for(int k=0; k < d; k++){
        float x = U[d*i+k];
        float y = v_sign * V[d*j+k];
        float diff = (x-y);
        sum_of_squares += diff * diff;
    }
    C[d*j_c+i] = sum_of_squares;
}


float* compute_cost_matrix(int d, float* h_U, float* h_V){
    /*
        Computes the cost matrix needed to compute the symmetrized Frobenius norm
        approximation criterion between matrices U and V.
    */
    int n = 2*d*d;
    int m = d*d;
    float *d_U; 
    float *d_V;
    float *d_C;
    cudaMalloc(&d_U, m*sizeof(float)); 
    cudaMalloc(&d_V, m*sizeof(float)); 
    cudaMalloc(&d_C, n*sizeof(float)); 
    cudaMemcpy(d_U, h_U, m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, m*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32,32,1);
    dim3 dimGrid(ceil((double) d / dimBlock.x), ceil((double) 2*d / dimBlock.y), 1);
    cost_matrix_kernel<<<dimGrid, dimBlock>>>(d, d_U, d_V, d_C);
    CUDA_CHECK;

    float *h_C = new float[n];
    cudaMemcpy(h_C, d_C, n*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_C);

    return h_C;
}
