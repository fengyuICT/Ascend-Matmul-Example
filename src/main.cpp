/**
* @file main.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>

#include <dirent.h>
#include <cassert>

#include "acl/acl.h"
//#include "kernel_select.h"

const char* op_type = "MatMulYF";
const char* kernel_name = "simple_matmul__kernel0";
const char* dummy_kernel_name = "simple_matmul_dummy__kernel0";

#define ACL_REQUIRES_OK(expr) \
    do { \
        auto __ret = (expr); \
        if (__ret != ACL_SUCCESS) { \
            return __ret; \
        } \
    } \
    while (0)

bool CheckDims(size_t n, size_t c, size_t h, size_t w) {
    if (n == 0 || c == 0 || h == 0 || w == 0) {
        std::cout << "[ERROR] All dim can't support 0" << std::endl;
        return false;
    }

    if(n != 1){
        std::cout << "[ERROR] Batch N only support 1" << std::endl;
        return false;
    }

    return true;
}

extern "C" aclError SelectAclopMatMul(int numInputs, const aclTensorDesc *const inputDesc[],
                                         int numOutputs, const aclTensorDesc *const outputDesc[],
                                         const aclopAttr *opAttr, aclopKernelDesc *aclopKernelDesc)
{
  const char* scheduleFlag = kernel_name;
  int32_t batch = 512; 
  aclopSetKernelArgs(aclopKernelDesc, scheduleFlag, 1, &batch, sizeof(int32_t));
  return ACL_SUCCESS;
}

void MatMulOnCPU(aclFloat16 *matrix_a, aclFloat16 *matrix_b, float *output, int M, int N, int K) {
  // out = [N/16, M, 16]
  // a = [K/16, M, 16]
  // b = [N/16, K, 16]
  int out_n = N/16, inner_n = 16;
  int out_k = K/16, inner_k = 16;
  for (int on = 0; on < out_n; on++) {
    for (int m = 0; m < M; m++) {
      for (int in = 0; in < inner_n; in++) {
        int idx = on*M*inner_n+m*inner_n+in;
        output[idx] = 0;
        for (int ok = 0; ok < out_k; ok++) {
          for (int ik = 0; ik < inner_k; ik++) {
            int real_k = ok*inner_k+ik;
            float a = aclFloat16ToFloat(matrix_a[ok*M*inner_k+m*inner_k+ik]);
            float b = aclFloat16ToFloat(matrix_b[on*K*inner_n+real_k*inner_n+in]);
            output[idx] += a*b;
          }
        }
      }
    }
  }
}

void DisplayResult(int64_t &M, int64_t &N, float *output, float* host_output) {
    for (int i = 0; i < M*N; ++i) {
      float dev_out = output[i];
      //float dev_out = aclFloat16ToFloat(output[i]);
      double diff = std::fabs(dev_out-host_output[i]);
      if (diff > 0.1) {
        std::cout << "Incorrect: " 
            << i << ", "
            << dev_out << ", " 
            << host_output[i] << ", "
            << diff << std::endl;
        return;
      }
    }
    std::cout << "Result Correct" << std::endl;
}

aclError MatMulTest(int64_t M, int64_t N, int64_t K)
{
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        std::cout << "[ERROR] Get run mode failed" << std::endl;
        return ACL_ERROR_BAD_ALLOC;
    }

    // input size is shape * sizeof(float16)
    int64_t size_matrix_a = M * K * 2;
    int64_t size_matrix_b = K * N * 2;
    int64_t size_output = M * N * sizeof(float);
    int hardware_limit = 16;

    int64_t shape_matrix_a[] = {K/hardware_limit, M, hardware_limit};
    int64_t shape_matrix_b[] = {K/hardware_limit, N, hardware_limit};
    int64_t shape_output[] = {N/hardware_limit, M, hardware_limit};

    aclFloat16 *matrix_a = new(std::nothrow) aclFloat16[M*K];
    if (matrix_a == NULL) {
        return ACL_ERROR_BAD_ALLOC;
    }
    aclFloat16 *matrix_b = new(std::nothrow) aclFloat16[K*N];
    if (matrix_b == NULL) {
        delete[]matrix_a;
        return ACL_ERROR_BAD_ALLOC;
    }
    float *output = new(std::nothrow) float[M*N];
    if (output == NULL) {
        delete[]matrix_a;
        delete[]matrix_b;
        return ACL_ERROR_BAD_ALLOC;
    }

    aclTensorDesc *input_desc[2];
    aclTensorDesc *output_desc[1];
    input_desc[0] = aclCreateTensorDesc(ACL_FLOAT16, 3, shape_matrix_a, ACL_FORMAT_ND);
    input_desc[1] = aclCreateTensorDesc(ACL_FLOAT16, 3, shape_matrix_b, ACL_FORMAT_ND);
    output_desc[0] = aclCreateTensorDesc(ACL_FLOAT, 3, shape_output, ACL_FORMAT_ND);

    //initilize data
    for (int i = 0; i < M*K; ++i) {
        matrix_a[i] = aclFloatToFloat16(1.0f);
    }

    for (int i = 0; i < K*N; ++i) {
        matrix_b[i] = aclFloatToFloat16(0.5f);
    }
    for (int i = 0; i <M*N; i++) {
      output[i] = 1.0f;
    }

    aclrtStream stream;
    ACL_REQUIRES_OK(aclrtCreateStream(&stream));

    void *devInput_a = nullptr;
    void *devInput_b = nullptr;
    void *devOutput = nullptr;

    ACL_REQUIRES_OK(aclrtMalloc(&devInput_a, size_matrix_a, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_REQUIRES_OK(aclrtMalloc(&devInput_b, size_matrix_b, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_REQUIRES_OK(aclrtMalloc(&devOutput, size_output, ACL_MEM_MALLOC_NORMAL_ONLY));

    aclrtMemcpyKind kindInput = ACL_MEMCPY_HOST_TO_DEVICE;
    if (runMode == ACL_DEVICE) {
        kindInput = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    ACL_REQUIRES_OK(aclrtMemcpy(devInput_a, size_matrix_a, matrix_a, size_matrix_a, kindInput));
    ACL_REQUIRES_OK(aclrtMemcpy(devInput_b, size_matrix_b, matrix_b, size_matrix_b, kindInput));

    aclDataBuffer *inputBuffer_a = aclCreateDataBuffer(devInput_a, size_matrix_a);
    aclDataBuffer *inputBuffer_b = aclCreateDataBuffer(devInput_b, size_matrix_b);
    aclDataBuffer *outputBuffer = aclCreateDataBuffer(devOutput, size_output);

    aclDataBuffer *inputs[] = {inputBuffer_a, inputBuffer_b};
    aclDataBuffer *outputs[] = {outputBuffer};

    ACL_REQUIRES_OK(
            aclopUpdateParams(op_type, 2, input_desc, 1, output_desc, nullptr));
    ACL_REQUIRES_OK(aclopExecuteV2(op_type, 2, input_desc, inputs, 1, output_desc, outputs, nullptr, stream));
    ACL_REQUIRES_OK(aclrtSynchronizeStream(stream));
    aclrtEvent start_event;
    aclrtEvent end_event;
    aclrtCreateEvent(&start_event);
    aclrtCreateEvent(&end_event);

    float elapse = 0;
    int num_iter = 1000;
    for (int i = 0; i < num_iter; i++) {
      aclrtRecordEvent(start_event, stream);
      ACL_REQUIRES_OK(aclopExecuteV2(op_type, 2, input_desc, inputs, 1, output_desc, outputs, nullptr, stream));
      aclrtRecordEvent(end_event, stream);
      ACL_REQUIRES_OK(aclrtSynchronizeStream(stream));
      float tmp_time = 0;
      aclrtEventElapsedTime(&tmp_time, start_event, end_event);
      elapse += tmp_time;
      aclrtResetEvent(start_event, stream);
      aclrtResetEvent(end_event, stream);
    }
    std::cout << "Matrix_" << M << "_" << N << "_" << K
        << " time=" << elapse/num_iter << "ms" << std::endl;

    aclrtDestroyEvent(start_event);
    aclrtDestroyEvent(end_event);
    ACL_REQUIRES_OK(aclrtDestroyStream(stream));

    aclrtMemcpyKind kindOutput = ACL_MEMCPY_DEVICE_TO_HOST;
    if (runMode == ACL_DEVICE) {
        kindOutput = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
    ACL_REQUIRES_OK(aclrtMemcpy(output, size_output, devOutput, size_output, kindOutput));

    float* host_output = (float*)malloc(M*N*sizeof(float));
    MatMulOnCPU(matrix_a, matrix_b, host_output, M, N, K);

    // display the result of this case
    DisplayResult(M, N, output, host_output);

    ACL_REQUIRES_OK(aclrtFree(devInput_a));
    ACL_REQUIRES_OK(aclrtFree(devInput_b));
    ACL_REQUIRES_OK(aclrtFree(devOutput));

    ACL_REQUIRES_OK(aclDestroyDataBuffer(inputBuffer_a));
    ACL_REQUIRES_OK(aclDestroyDataBuffer(inputBuffer_b));
    ACL_REQUIRES_OK(aclDestroyDataBuffer(outputBuffer));

    for (auto desc : input_desc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto desc : output_desc) {
        aclDestroyTensorDesc(desc);
    }

    delete[]matrix_a;
    delete[]matrix_b;
    delete[]output;

    return ACL_SUCCESS;
}

bool ReadBytesFromBinaryFile(const char *file_name, char **buffer, int &length)
{
    if (file_name == nullptr) {
        std::cout << "File is nullptr, parameter error." << std::endl;
        return false;
    }

    std::string real_path = file_name;
    real_path = "kernel_meta/" + real_path;

    std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("Read file %s failed.", file_name);
        return false;
    }

    length = static_cast<int>(file.tellg());
    if (length <= 0) {
        printf("File length <= 0");
        file.close();
        return false;
    }

    file.seekg(0, std::ios::beg);

    *buffer = new(std::nothrow) char[length]();
    if (buffer == nullptr) {
        printf("New an object failed");
        file.close();
        return false;
    }

    file.read(*buffer, length);
    file.close();
    return true;
}

void Deallocator(void *data, size_t size)
{
    auto addr = reinterpret_cast<char *>(data);
    delete[]addr;
}

int main(int argc, char **argv)
{
    //if (argc != 2) {
    //    printf("argc number is not 1!\n");
    //    return -1;
    //}

    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::cout << "Acl init failed." << std::endl;
        return -1;
    }

    const char file_1[] = "simple_matmul.o";
    //const char file_2[] = "simple_matmul_dummy.o";

    char *buffer = nullptr;
    int length = 0;

    aclrtSetDevice(0);
    uint32_t device_count;
    aclrtGetDeviceCount(&device_count);
    std::cout << "aclrtGetDeviceCount: " << device_count << std::endl;

    if (aclopRegisterCompileFunc(op_type, SelectAclopMatMul)) {
      std::cout << "aclopRegisterCompileFunc failed." << std::endl;
      return -1;
    }

    ReadBytesFromBinaryFile(file_1, &buffer, length);
    if (aclopCreateKernel(op_type, kernel_name, kernel_name,
                      buffer, length, ACL_ENGINE_AICORE, Deallocator)) {
        std::cout << "aclopCreateKernel failed." << std::endl;
        return -1;
    }

    //ReadBytesFromBinaryFile(file_2, &buffer, length);
    //aclopCreateKernel(op_type, dummy_kernel_name, dummy_kernel_name,
    //                  buffer, length, ACL_ENGINE_AICORE, Deallocator);

    int M = 16;
    int N = 16;
    int K = 16;
    if (M <= 0 ) {
            std::cout << "[ERROR] The shape is invalid that N must be positive integer."
                      << std::endl;
            return -1;
    }

    //main process
    auto result_type = MatMulTest(M, N, K);
    if (result_type != ACL_SUCCESS) {
        std::cout << "[ERROR] MatMul test failed: " << result_type << std::endl;
        return -1;
    }

    aclrtResetDevice(0);

    if (aclFinalize() != ACL_SUCCESS) {
      std::cout << "[ERROR] Finalize acl failed." << std::endl;
      return -1;
    }
    return 0;
}
