/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

#include "CL/cl.h"

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include <sys/time.h>

namespace tflite {
namespace multithreaded_ops {

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType TfLitePadding2EigenPadding(TfLitePadding padding) {
    switch (padding) {
      case kTfLitePaddingValid:
        return Eigen::PADDING_VALID;
      case kTfLitePaddingSame:
        return Eigen::PADDING_SAME;
      case kTfLitePaddingUnknown:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const Eigen::ThreadPoolDevice& device, const T* input_data,
                  T* im2col_buffer, int input_batches, int input_height,
                  int input_width, int input_depth, const T* filter_data,
                  int filter_height, int filter_width, int filter_count,
                  int stride_rows, int stride_cols, int pad_width,
                  int pad_height, TfLitePadding padding, T* output_data,
                  int output_height, int output_width) {
    const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                stride_rows == 1 && stride_cols == 1);
    if (is_1x1_kernel) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, input_batches * conv_width, filter_count);
      ConstEigenMatrix input(input_data, input_batches * conv_width,
                             input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else if (filter_height == input_height && filter_width == input_width &&
               pad_width == 0 && pad_height == 0) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter_width * filter_height * input_depth;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, 1, filter_count);
      ConstEigenMatrix input(input_data, 1, k);
      ConstEigenMatrix filter(filter_data, k, filter_count);
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else {
      EigenTensor output(output_data, input_batches, output_height,
                         output_width, filter_count);
      ConstEigenTensor input(input_data, input_batches, input_height,
                             input_width, input_depth);
      ConstEigenTensor filter(filter_data, filter_height, filter_width,
                              input_depth, filter_count);
      output.device(device) =
          Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                    TfLitePadding2EigenPadding(padding));
    }
  }
};

inline void Conv(const Eigen::ThreadPoolDevice& device, const float* input_data,
                 const Dims<4>& input_dims, const float* filter_data,
                 const Dims<4>& filter_dims, const float* bias_data,
                 const Dims<4>& bias_dims, int stride_width, int stride_height,
                 int pad_width, int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(device, input_data, im2col_data, batches, input_height,
               input_width, input_depth, filter_data, filter_height,
               filter_width, output_depth, stride_height, stride_width,
               pad_height, pad_width, padding, output_data, output_height,
               output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
}

static cl_kernel kernelconvFilterAndImageCache = NULL;
static cl_kernel kernelmatmulInputCache = NULL;
size_t convWgHeight = CONV_WG_HEIGHT;
size_t convWgWidth = CONV_WG_WIDTH;
size_t matmulWgHeight = MATMUL_WG_HEIGHT;
size_t matmulWgWidth = 4*MATMUL_WG_HEIGHT;

inline void OpenCLConv(const float* input_data, int input_size,
          const float* filter_data, int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          int* dim_sizes, int* dim_strides,
          float output_activation_min, float output_activation_max,
          cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {
  cl_mem d_input = cl_mem_arr[0];
  cl_mem d_filter = cl_mem_arr[1];
  cl_mem d_output = cl_mem_arr[3];
  cl_mem d_dim_sizes = cl_mem_arr[4];
  cl_mem d_dim_strides = cl_mem_arr[5];

  cl_event event_runkernel;
  cl_event event_mapinput;
  cl_event event_mapfilter;
  cl_event event_mapoutput;
  cl_event event_unmapinput;
  cl_event event_unmapfilter;
  cl_event event_unmapoutput;
  cl_event event_writedimsizes;
  cl_event event_writedimstrides;

  cl_int err;
  
  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  int numchannel = dim_sizes[0];
  int addslot = (4-(numchannel%4))%4;
  numchannel = numchannel + addslot;
  input_size = (input_size/dim_sizes[0])*numchannel;
  filter_size = (filter_size/dim_sizes[0])*numchannel;

  float *inputfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_input,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              input_size*sizeof(float),
              0, NULL, NULL, &err);
  float *filterfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_filter,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              filter_size*sizeof(float),
              0, NULL, NULL, &err);

  for(int i = 0,i2 = 0; i < input_size; i+=numchannel,i2+=dim_sizes[0]) {
    for(int j = 0; j < dim_sizes[0]; j++) {
      inputfloat[i+j] = input_data[i2+j];
    }
    for(int j = dim_sizes[0]; j < numchannel; j++) {
      inputfloat[i+j] = 0.0;
    }
  }
  for(int i = 0,i2 = 0; i < filter_size; i+=numchannel,i2+=dim_sizes[0]) {
    for(int j = 0; j < dim_sizes[0]; j++) {
      filterfloat[i+j] = filter_data[i2+j];
    }
    for(int j = dim_sizes[0]; j < numchannel; j++) {
      filterfloat[i+j] = 0.0;
    }
  }

  clEnqueueUnmapMemObject(queue,d_input,(void *) inputfloat,0, NULL, &event_unmapinput);
  clEnqueueUnmapMemObject(queue,d_filter,(void *) filterfloat,0, NULL, &event_unmapfilter);

  dim_strides[1] = (dim_strides[1]/dim_sizes[0])*numchannel;
  dim_strides[2] = (dim_strides[2]/dim_sizes[0])*numchannel;
  dim_strides[3] = (dim_strides[3]/dim_sizes[0])*numchannel;
  dim_strides[5] = (dim_strides[5]/dim_sizes[0])*numchannel;
  dim_strides[6] = (dim_strides[6]/dim_sizes[0])*numchannel;
  dim_strides[7] = (dim_strides[7]/dim_sizes[0])*numchannel;
  dim_sizes[0] = numchannel;
  dim_sizes[4] = numchannel;

  int d_output_depth = (((output_depth-1)/4+1)*4);

  dim_strides[13] = (dim_strides[13]/dim_sizes[12])*d_output_depth/4;
  dim_strides[14] = (dim_strides[14]/dim_sizes[12])*d_output_depth/4;
  dim_strides[15] = (dim_strides[15]/dim_sizes[12])*d_output_depth/4;

  err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
                                 16*sizeof(int), dim_sizes, 0, NULL, &event_writedimsizes);
  err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
                                 16*sizeof(int), dim_strides, 0, NULL, &event_writedimstrides);

  clFinish(queue);

  if((dim_sizes[6] == 1) && (dim_sizes[5] == 1) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
  
    int m_cols = dim_sizes[0];
    int m_rows = dim_sizes[1]*dim_sizes[2]*dim_sizes[3];
    int n_batch = dim_sizes[7];
    int bias_stride = dim_strides[8];

    err  = clSetKernelArg(kernelmatmulInputCache, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernelmatmulInputCache, 1, sizeof(cl_mem), &d_filter);
    err  = clSetKernelArg(kernelmatmulInputCache, 2, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernelmatmulInputCache, 3, sizeof(int), &m_rows);
    err  = clSetKernelArg(kernelmatmulInputCache, 4, sizeof(int), &m_cols);
    err  = clSetKernelArg(kernelmatmulInputCache, 5, sizeof(int), &n_batch);
    
    const size_t local[2] = { matmulWgHeight, matmulWgWidth };
    const size_t global[2] = { (size_t) ((d_output_depth/4-1)/matmulWgHeight+1)*matmulWgHeight, (size_t) ((output_height*output_width*batches-1)/matmulWgWidth+1)*matmulWgWidth };

    err = clEnqueueNDRangeKernel(queue, kernelmatmulInputCache, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    __android_log_print(ANDROID_LOG_INFO, "OpenCLDebug", "Error Kernel: %d", err);

    cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_output,
            CL_TRUE,
            CL_MAP_READ,
            0,
            output_size/output_depth*d_output_depth*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < output_size/output_depth; i++) {
      for(int j = 0; j < output_depth; j++) {
        output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
      }
    }

    clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
    clFinish(queue);
  }
  else if((dim_sizes[6] < convWgHeight) && (dim_sizes[5] < convWgHeight) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    int xsize = ((output_width-1)/convWgWidth+1)*convWgWidth;
    int ysize = ((output_height-1)/convWgHeight+1)*convWgHeight;

    err  = clSetKernelArg(kernelconvFilterAndImageCache, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 1, sizeof(cl_mem), &d_filter);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 2, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 3, sizeof(int), &stride_width);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 4, sizeof(int), &stride_height);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 5, sizeof(int), &pad_width);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 6, sizeof(int), &pad_height);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 7, sizeof(int), &xsize);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 8, sizeof(int), &ysize);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 9, sizeof(cl_mem), &d_dim_sizes);
    err  = clSetKernelArg(kernelconvFilterAndImageCache, 10, sizeof(cl_mem), &d_dim_strides);

    const size_t local[2] = { convWgHeight, convWgWidth };
    const size_t global[2] = { (size_t) ysize*batches, (size_t) xsize*d_output_depth/4 };
    
    err = clEnqueueNDRangeKernel(queue, kernelconvFilterAndImageCache, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    __android_log_print(ANDROID_LOG_INFO, "OpenCLDebug", "Error Kernel: %d", err);

    cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_output,
            CL_TRUE,
            CL_MAP_READ,
            0,
            output_size/output_depth*d_output_depth*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < output_size/output_depth; i++) {
      for(int j = 0; j < output_depth; j++) {
        output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
      }
    }

    clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
    clFinish(queue);
  }
}

inline void ConvOpenCL(const Eigen::ThreadPoolDevice& device, const float* input_data,
                 const Dims<4>& input_dims, const float* filter_data, const float* preprocessed_filter_data,
                 const Dims<4>& filter_dims, const float* bias_data,
                 const Dims<4>& bias_dims, int stride_width, int stride_height,
                 int pad_width, int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims,
                 cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {
  if((kernelmatmulInputCache == NULL) || (kernelconvFilterAndImageCache == NULL)) {
    kernelmatmulInputCache = clCreateKernel(program, "matmulInputCache", NULL);
    kernelconvFilterAndImageCache = clCreateKernel(program, "convFilterAndImageCache", NULL);
  }
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  int* sizes;
  int* strides;

  sizes = (int*)malloc(16*sizeof(int));
  strides = (int*)malloc(16*sizeof(int));

  //input
  sizes[0] = input_dims.sizes[0];
  sizes[1] = input_dims.sizes[1];
  sizes[2] = input_dims.sizes[2];
  sizes[3] = input_dims.sizes[3];
  strides[0] = input_dims.strides[0];
  strides[1] = input_dims.strides[1];
  strides[2] = input_dims.strides[2];
  strides[3] = input_dims.strides[3];

  //filter
  sizes[4] = filter_dims.sizes[0];
  sizes[5] = filter_dims.sizes[1];
  sizes[6] = filter_dims.sizes[2];
  sizes[7] = filter_dims.sizes[3];
  strides[4] = filter_dims.strides[0];
  strides[5] = filter_dims.strides[1];
  strides[6] = filter_dims.strides[2];
  strides[7] = filter_dims.strides[3];

  //bias
  sizes[8] = bias_dims.sizes[0];
  sizes[9] = bias_dims.sizes[1];
  sizes[10] = bias_dims.sizes[2];
  sizes[11] = bias_dims.sizes[3];
  strides[8] = bias_dims.strides[0];
  strides[9] = bias_dims.strides[1];
  strides[10] = bias_dims.strides[2];
  strides[11] = bias_dims.strides[3];

  //output
  sizes[12] = output_dims.sizes[0];
  sizes[13] = output_dims.sizes[1];
  sizes[14] = output_dims.sizes[2];
  sizes[15] = output_dims.sizes[3];
  strides[12] = output_dims.strides[0];
  strides[13] = output_dims.strides[1];
  strides[14] = output_dims.strides[2];
  strides[15] = output_dims.strides[3];

  int input_size = batches*input_width*input_height*input_depth;
  int filter_size = input_depth*output_depth*filter_width*filter_height;
  int bias_size = output_depth;
  int output_size = batches*output_width*output_height*output_depth;
  int im2col_size = im2col_dims.sizes[0]*im2col_dims.sizes[1]*im2col_dims.sizes[2]*im2col_dims.sizes[3];

  if((sizes[6] == 1) && (sizes[5] == 1) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program, cl_mem_arr);

    optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
  }
  else if((sizes[6] < 8) && (sizes[5] < 8) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program, cl_mem_arr);

    optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
  }
  else {
      Conv(device, input_data, input_dims,
        preprocessed_filter_data, filter_dims,
        bias_data, bias_dims,
        stride_width, stride_height, 
        pad_width, pad_height, 
        padding,
        output_activation_min, output_activation_max,
        output_data, output_dims,
        im2col_data, im2col_dims);
  }

  free(sizes);
  free(strides);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
