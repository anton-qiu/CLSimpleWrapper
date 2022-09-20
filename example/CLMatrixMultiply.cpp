
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <string>
#include <stdlib.h> 
#include <cmath>

#include "CLSimpleWrapper.h"

// currently simplify platform and device selection using this
#define PLATFORM_ID 1
#define DEVICE_ID -1

//#define MATRIX_TYPE_DOUBLE

#ifdef MATRIX_TYPE_DOUBLE
    typedef double MATRIX_TYPE;
#else
    typedef int MATRIX_TYPE;
#endif

#define MATRIX_DIMENSION	1000
#define MAX_VAL 1000
#define MIN_VAL 1

// square MatrixMultiplication: A*B=C, where each matrix is of dimension MxM
//  this example linearize the matrix into array.
//  it means: A[i][j] = a[(i*M)+j]
//  each of the work item will compute the result of one C[col][row]
//  this function does not use matrix transpose optimization.
std::string ClSrcStrMulMatInt =
"__kernel void multiplyMatrices(__global int* a,\
    __global int* b,\
    __global int* c,\
    const int M)\
{\
    int colIndex = get_global_id(0);\
    int rowIndex = get_global_id(1);\
    int index = (rowIndex * M) + colIndex;\
    int sum = 0;\
    for ( int k = 0; k < M; k++ )\
    {\
        sum += a[rowIndex * M + k] * b[k * M + colIndex];\
    }\
    c[index] = sum;\
}";

std::string ClSrcStrMulMatDouble =
"__kernel void multiplyMatrices(__global double* a,\
    __global double* b,\
    __global double* c,\
    const int M)\
{\
    int colIndex = get_global_id(0);\
    int rowIndex = get_global_id(1);\
    int index = (rowIndex * M) + colIndex;\
    double sum = 0;\
    for ( int k = 0; k < M; k++ )\
    {\
        sum += a[rowIndex * M + k] * b[k * M + colIndex];\
    }\
    c[index] = sum;\
}";

// square MatrixMultiplication: A*B=C, where each matrix is of dimension MxM
//  it means: A[i][j] = a[(i*M)+j]
//  each of the work item will compute the result of one C[col][row]
//  this function requires the matrix B being transposed before linearized and pass to this function.
std::string ClSrcStrMulMatIntTrans =
"__kernel void multiplyMatricesTrans(__global int* a,\
    __global int* b,\
    __global int* c,\
    const int M)\
{\
    int colIndex = get_global_id(0);\
    int rowIndex = get_global_id(1);\
    int index = (rowIndex * M) + colIndex;\
    int sum = 0;\
    for ( int k = 0; k < M; k++ )\
    {\
        sum += a[rowIndex * M + k] * b[colIndex * M + k];\
    }\
    c[index] = sum;\
}";

std::string ClSrcStrMulMatDoubleTrans =
"__kernel void multiplyMatricesTrans(__global double* a,\
    __global double* b,\
    __global double* c,\
    const int M)\
{\
    int colIndex = get_global_id(0);\
    int rowIndex = get_global_id(1);\
    int index = (rowIndex * M) + colIndex;\
    double sum = 0;\
    for ( int k = 0; k < M; k++ )\
    {\
        sum += a[rowIndex * M + k] * b[colIndex * M + k];\
    }\
    c[index] = sum;\
}";

void parallelOpenCLMatrixMultTrans(MATRIX_TYPE* matrixA, MATRIX_TYPE* matrixBTrans, MATRIX_TYPE* matrixResult)
{
    CLSimpleWrapper cl_wrapper;

    cl_wrapper.initOpenCL(PLATFORM_ID, DEVICE_ID, false);   // use first platform and device
    //cl_wrapper.createCLCommandQueue();
#ifdef MATRIX_TYPE_DOUBLE
    cl_wrapper.createCLKernel(ClSrcStrMulMatDoubleTrans, "multiplyMatricesTrans");   // integer matrix multiplication.
#else
    cl_wrapper.createCLKernel(ClSrcStrMulMatIntTrans, "multiplyMatricesTrans");   // integer matrix multiplication.
#endif
    cl_wrapper.setKernelBufferArg(0, (void*)matrixA, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_wrapper.setKernelBufferArg(1, (void*)matrixBTrans, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_wrapper.setKernelBufferArg(2, nullptr, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_int matrix_dimension = MATRIX_DIMENSION;
    cl_wrapper.setKernelArg(3, &matrix_dimension, sizeof(cl_int));

    size_t global_item_size[2] = { MATRIX_DIMENSION , MATRIX_DIMENSION };   // size of matrix
    cl_wrapper.executeKernel(2, global_item_size, NULL);
    cl_wrapper.readBuffer(matrixResult, 2, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
}

void parallelOpenCLMatrixMult(MATRIX_TYPE* matrixA, MATRIX_TYPE* matrixB, MATRIX_TYPE* matrixResult)
{
    CLSimpleWrapper cl_wrapper;

    cl_wrapper.initOpenCL(PLATFORM_ID, DEVICE_ID, false);   // use first platform and device
#ifdef MATRIX_TYPE_DOUBLE
    cl_wrapper.createCLKernel(ClSrcStrMulMatDouble, "multiplyMatrices");   // integer matrix multiplication.
#else
    cl_wrapper.createCLKernel(ClSrcStrMulMatInt, "multiplyMatrices");   // integer matrix multiplication.
#endif
    cl_wrapper.setKernelBufferArg(0, (void*)matrixA, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_wrapper.setKernelBufferArg(1, (void*)matrixB, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_wrapper.setKernelBufferArg(2, nullptr, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    cl_int matrix_dimension = MATRIX_DIMENSION;
    cl_wrapper.setKernelArg(3, &matrix_dimension, sizeof(cl_int));

    size_t global_item_size[2] = { MATRIX_DIMENSION , MATRIX_DIMENSION };   // size of matrix
    cl_wrapper.executeKernel(2, global_item_size, NULL);
    cl_wrapper.readBuffer(matrixResult, 2, MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
}

void clearMatrix(MATRIX_TYPE* matrix, std::string mat_name)
{
    std::cout << "clearing matrix : " << mat_name << std::endl;
    for ( int i = 0; i < MATRIX_DIMENSION; i++ )
    {
        for ( int j = 0; j < MATRIX_DIMENSION; j++ )
        {
            matrix[(i * MATRIX_DIMENSION) + j] = 0;
        }
    }
}

int main(int argc, char* argv[])
{
    MATRIX_TYPE* matrixA;
    MATRIX_TYPE* matrixB;
    MATRIX_TYPE* matrixBTrans;	// transposition of Matrix B
    MATRIX_TYPE* matrixResult;

    // Initialize matrix with random value

    matrixA = (MATRIX_TYPE*)malloc(MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    matrixB = (MATRIX_TYPE*)malloc(MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    matrixBTrans = (MATRIX_TYPE*)malloc(MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));
    matrixResult = (MATRIX_TYPE*)malloc(MATRIX_DIMENSION * MATRIX_DIMENSION * sizeof(MATRIX_TYPE));

    for ( int i = 0; i < MATRIX_DIMENSION; i++ )
    {
        for ( int j = 0; j < MATRIX_DIMENSION; j++ )
        {
            //matrixA[i][j] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
            //matrixB[i][j] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
            //matrixBTrans[j][i] = matrixB[i][j];	// transposition of Matrix B
            //matrixResult[i][j] = 0;	// init result to zero...
            //matrixResultReference[i][j] = 0;	// init result to zero...
            matrixA[(i * MATRIX_DIMENSION) + j] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
            matrixB[(i * MATRIX_DIMENSION) + j] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
            matrixBTrans[(j * MATRIX_DIMENSION) + i] = matrixB[(i * MATRIX_DIMENSION) + j];	// transposition of Matrix B
            matrixResult[(i * MATRIX_DIMENSION) + j] = 0;	// init result to zero...
        }
    }

    std::cout << "------------------------------------------------------------------------ \n";
    std::cout << "  Calculate Matrix Multiplication using OpenCL (no transpose): \n";
    std::cout << "------------------------------------------------------------------------ \n";
    std::cout << "Starting OpenCL (no transpose)... " << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    parallelOpenCLMatrixMult(matrixA, matrixB, matrixResult);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Parallel OpenCL (no transpose) ended. " << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    clearMatrix(matrixResult, "matrixResult");

    std::cout << "------------------------------------------------------------------------ \n";
    std::cout << "  Calculate Matrix Multiplication using OpenCL (with transpose): \n";
    std::cout << "------------------------------------------------------------------------ \n";
    std::cout << "Starting OpenCL (with transpose)... " << std::endl;
    start = std::chrono::high_resolution_clock::now();
    parallelOpenCLMatrixMultTrans(matrixA, matrixBTrans, matrixResult);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "Parallel OpenCL (with transpose) ended. " << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";
    clearMatrix(matrixResult, "matrixResult");

}
