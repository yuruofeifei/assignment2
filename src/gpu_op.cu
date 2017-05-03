#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

__global__ void array_set_kernel(float *data, float value, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    data[i] = value;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < arr->ndim; i++) {
    size *= arr->shape[i];
  }
  array_set_kernel<<<1, 1024>>>((float *)arr->data, value, size);
  return 0;
}

__global__ void broadcast_to_kernel(const float *input, float *output, int64_t size, int64_t nrows) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < nrows; i += stride) {
    memcpy(output + i * size, input, sizeof(float) * size);
  }
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < input->ndim; i++) {
    size *= input->shape[i];
  }
  broadcast_to_kernel<<<1, 1024>>>((const float *)input->data, (float *)output->data, size, output->shape[0]);
  return 0;
}

__global__ void gpu_reduce_sum_axis_zero_kernel(const float *input, float *output, int64_t size, int64_t nrows) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    float v = 0.0;
    for (int j = 0; j < nrows; j++) {
      v += input[j * size + i];
    }
    output[i] = v; 
  }
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_reduce_sum_axis_zero_kernel<<<1, 1024>>>((const float *)input->data, (float *)output->data, size, input->shape[0]);
  return 0;
}

__global__ void gpu_matrix_add(const float *a, const float *b, float *output, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    output[i] = a[i] + b[i];
  }
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_matrix_add<<<1, 1024>>>((const float *)matA->data, (const float *)matB->data, (float *)output->data, size);
  return 0;
}

__global__ void gpu_matrix_add_const(const float *input, float *output, int64_t size, float v) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    output[i] = input[i] + v;
  }
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_matrix_add_const<<<1, 1024>>>((const float *)input->data, (float *)output->data, size, val);
  return 0;
}

__global__ void gpu_matrix_multiply(const float *a, const float *b, float *output, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    output[i] = a[i] * b[i];
  }
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_matrix_multiply<<<1, 1024>>>((const float *)matA->data, (const float *)matB->data, (float *)output->data, size);
  return 0;
}

__global__ void gpu_matrix_multiply_const(const float *input, float *output, int64_t size, float v) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    output[i] = input[i] * v;
  }
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_matrix_multiply_const<<<1, 1024>>>((const float *)input->data, (float *)output->data, size, val);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasOperation_t trans_a = CUBLAS_OP_N, trans_b = CUBLAS_OP_N;
  int m = matC->shape[0], k = matA->shape[1], n = matC->shape[1];
  const float alpha = 1.0;
  const float beta = 0.0;
  if (transposeA) {
    trans_a = CUBLAS_OP_T;
    k = matA->shape[0];
  }
  if (transposeB) {
    trans_b = CUBLAS_OP_T;
  }
  cublasSgemm(handle, trans_b, trans_a,
              n, m, k, &alpha,
              (const float *)matB->data, transposeB ? k : n,
              (const float *)matA->data, transposeA ? m : k,
              &beta, (float *)matC->data, n);
  
  return 0;
}

__global__ void gpu_relu_kernel(const float *input, float *output, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    output[i] = max(0.0f, input[i]);
  }
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_relu_kernel<<<1, 1024>>>((const float*)input->data, (float *)output->data, size);
  return 0;
}

__global__ void gpu_relu_gradient_kernel(const float *grad, const float *input, float *output, int64_t size) {
  int id = threadIdx.x;
  int stride = blockDim.x;
  for (int i = id; i < size; i += stride) {
    if (input[i] >= 0.0) {
      output[i] = grad[i];
    } else {
      output[i] = 0.0;
    }
  }
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  int64_t size = 1;
  for (int i = 0; i < output->ndim; i++) {
    size *= output->shape[i];
  }
  gpu_relu_gradient_kernel<<<1, 1024>>>((const float *)in_grad->data, (const float*)input->data, (float *)output->data, size);  
  return 0;
}

__global__ void matrix_softmax_kernel(int nrow, int ncol,
                                      const float *input,
                                      float *output) {
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input += y * ncol;
  float maxval = *input;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input[x]);
  }
  output += y * ncol;
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    output[x] = exp(input[x] - maxval);
    sum += output[x];
  }
  for (int x = 0; x < ncol; ++x) {
    output[x] /= sum;
  }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  dim3 threads;
  if (input->shape[0] <= 1024) {
    threads.x = input->shape[0];
  } else {
    threads.x = 1024;
    threads.y = (input->shape[0] + 1023) / 1024;
  }
  matrix_softmax_kernel<<<1, threads>>>(input->shape[0], input->shape[1], (const float *)input->data, (float *)output->data);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
