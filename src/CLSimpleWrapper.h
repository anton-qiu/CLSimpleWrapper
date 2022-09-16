#pragma once

#include <vector>
#include <string>

#define CL_TARGET_OPENCL_VERSION 120
//#define _CL_CPP   // not using OpenCL C++ binding for now

#include <CL/opencl.h>
#ifdef _CL_CPP
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/opencl.hpp"	// if using C++ binding
#endif

class CLSimpleWrapper
{
public:
    CLSimpleWrapper();
    ~CLSimpleWrapper();

    void initOpenCL(int platformId = -1, int deviceId = -1, bool is_list_only = true);

    cl_int createCLKernel(std::string& kernel_source_str, std::string prog_name);

    cl_int createCLKernelFromFile(std::string kernel_file_path, std::string prog_name);

    // note: we assume that args created in order
    void setKernelBufferArg(unsigned int index, void* buffer, size_t len);

    // generic implementation for setting OpenCL primitive type argument.
    void setKernelArg(unsigned int index, const void* buffer, const size_t len);

    // note: we assume that args created in order, caller is responsible to allocate the memory for outData.
    void readBuffer(void* outData, size_t index, size_t len);


    // let caller to have control the global item size and local item size
    void executeKernel(size_t workSize,     // size of work item: i.e. size of globalItemSize[] array.
        size_t* globalItemSize,
        size_t* localItemSize);// Divide work items into groups of localItemSize

private:
    std::string GetPlatformName(cl_platform_id id);

    std::string GetDeviceName(cl_device_id id);

    void clear();

    void checkCLError(cl_int error, std::string err_msg = "");


    cl_device_id m_device;
    cl_context m_context;

    cl_kernel m_kernel;
    cl_program m_program;
    cl_command_queue m_cmdQueue;

    std::vector<cl_mem > m_args;

};


