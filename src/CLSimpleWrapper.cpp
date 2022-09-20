
#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <string>
#include <fstream>
#include <stdlib.h> 

#include "CLSimpleWrapper.h"


CLSimpleWrapper::CLSimpleWrapper()
    : m_device(nullptr),
    m_context(nullptr),
    m_kernel(nullptr),
    m_program(nullptr),
    m_cmdQueue(nullptr)
{

}

CLSimpleWrapper::~CLSimpleWrapper()
{
    clear();
}

void CLSimpleWrapper::initOpenCL(int platformId, int deviceId, bool is_list_only)
{
    cl_int error = CL_SUCCESS;
    cl_uint platformIdCount = 0;
    std::vector<cl_device_id> deviceIds;	// init: empty result
    error = clGetPlatformIDs(0, nullptr, &platformIdCount);

    if ( platformIdCount == 0 )
    {
        std::cerr << "No OpenCL platform found" << std::endl;
        return;
    }
    else
    {
        std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
    }

    if ( 0 == platformIdCount )
    {
        std::cerr << "No openCL Platform! " << std::endl;
        std::exit(1);
    }

    // first get all the platforms
    std::vector<cl_platform_id> platformIds(platformIdCount);
    error = clGetPlatformIDs(platformIdCount, platformIds.data(), nullptr);

    // print all the platform and devices
    for ( cl_uint i = 0; i < platformIdCount; ++i )
    {
        std::cout << "\t" << i << ". Platform Name : " << GetPlatformName(platformIds[i]) << std::endl;

        cl_uint deviceIdCount = 0;
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, 0, nullptr,
            &deviceIdCount);
        if ( deviceIdCount == 0 )
        {
            std::cerr << "No OpenCL devices found" << std::endl;
            return;
        }
        else
        {
            std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
        }

        std::vector<cl_device_id> deviceIds(deviceIdCount);
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceIdCount,
            deviceIds.data(), nullptr);

        for ( int j = 0; j < deviceIdCount; j++ )
        {
            std::string deviceName = GetDeviceName(deviceIds[j]);
            std::cout << "\t\t" << i << "." << j << " Device Name : " << deviceName << std::endl;
        }
    }

    if ( is_list_only )
    {
        // list only, without selecting platform and device, return empty device
        return;
    }

    // select platform
    cl_platform_id platform;
    if ( platformId < 0 )
    {
        platform = platformIds[0];	// get the first platform.
    }
    else
    {
        platform = platformIds[platformId];
    }

    const cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds[platformId]),
        0, 0
    };

    std::string platformName = GetPlatformName(platform);
    std::cout << "\t Selected Platform Name : " << platformName << std::endl;

    // select device
    cl_uint deviceIdCount = 0;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr,
        &deviceIdCount);

    deviceIds.resize(deviceIdCount);	// allocate the space in vector, clGetDeviceIDs() take in as pointer.
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceIdCount,
        deviceIds.data(), nullptr);

    if ( deviceId < 0 )
    {
        m_device = deviceIds[0];	// get the first device.
    }
    else
    {
        m_device = deviceIds[deviceId];
    }

    std::string deviceName = GetDeviceName(m_device);
    std::cout << "\t Selected Device Name : " << deviceName << std::endl;

    //cl_context context = clCreateContext(g_contextProperties, devices.second.size(),
    //	devices.second.data(), nullptr, nullptr, &error);	// create context will all device within the platform
    m_context = clCreateContext(contextProperties, 1,
        &m_device, nullptr, nullptr, &error);	// create context only with the selected device
    checkCLError(error, "Create Context Failed");
    std::cout << "Context created" << std::endl;

    m_cmdQueue = clCreateCommandQueue(m_context, m_device, 0, &error);
    checkCLError(error, "Fail to create command queue");

    std::cout << "Command Queue created" << std::endl;

    return;
}

cl_int CLSimpleWrapper::createCLKernel(std::string& kernel_source_str, std::string prog_name)
{
    cl_int error = CL_SUCCESS;

    const char* src = kernel_source_str.c_str();
    m_program = clCreateProgramWithSource(m_context, 1,
        &src, NULL, &error);
    if ( error != CL_SUCCESS )
    {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(m_program, m_device,
            CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        std::cout << "Build Error: " << buffer << std::endl;
        clReleaseProgram(m_program);  // do we need to clear?
        m_program = nullptr;
        return error;
    }

    error = clBuildProgram(m_program, 1, &m_device, NULL, NULL, NULL);
    checkCLError(error, "Program Build Failed");

    m_kernel = clCreateKernel(m_program, prog_name.c_str(), &error);
    checkCLError(error, "Create Kernel Failed");

    return error;
}

cl_int CLSimpleWrapper::createCLKernelFromFile(std::string kernel_file_path, std::string prog_name)
{
    std::ifstream ifs(kernel_file_path);

    std::string kernel_source_str = ifs ? std::string(std::istreambuf_iterator<char>(ifs), (std::istreambuf_iterator<char>())) : "";

    return createCLKernel(kernel_source_str, prog_name);
}

// note: we assume that args created in order
void CLSimpleWrapper::setKernelBufferArg(unsigned int index, void* buffer, size_t len)
{
    cl_int error = CL_SUCCESS;
    cl_mem dev_buffer = nullptr;
    if ( nullptr == buffer )
    {
        dev_buffer = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY,   // for output
            len, NULL, &error);
        checkCLError(error, "Create Buffer Failed");
    }
    else
    {
        dev_buffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            len, buffer, &error);
        checkCLError(error, "Create Buffer Failed");
        
        //dev_buffer = clCreateBuffer(m_context, CL_MEM_READ_ONLY,
        //    len, nullptr, &error);
        //checkCLError(error, "Create Buffer Failed");
        //error = clEnqueueWriteBuffer(m_cmdQueue, dev_buffer, CL_TRUE, 0,
        //    len, buffer, 0, NULL, NULL);
        //checkCLError(error, "Enqueue Write Buffer Failed");
    }



    m_args.push_back(dev_buffer);   // keep the buffers to be cleared later

    setKernelArg(index, &dev_buffer, sizeof(cl_mem));
}

// generic implementation for setting OpenCL primitive type argument.
void CLSimpleWrapper::setKernelArg(unsigned int index, const void* buffer, const size_t len)
{
    cl_int error = CL_SUCCESS;
    error = clSetKernelArg(m_kernel, index,
        len, // size of the argument type in buffer.
        buffer);
    checkCLError(error, "Set Kernel Arg Failed");
}

// note: we assume that args created in order, caller is responsible to allocate the memory for outData.
void CLSimpleWrapper::readBuffer(void* outData, size_t index, size_t len)
{
    cl_int error = CL_SUCCESS;

    //outData = (int*)malloc(len); We are using blocking read here
    error = clEnqueueReadBuffer(m_cmdQueue, m_args[index], CL_TRUE, 0,
        len, outData, 0, NULL, NULL);
    checkCLError(error, "Enqueue Read Buffer Failed");
}

// let caller to have control the global item size and local item size
void CLSimpleWrapper::executeKernel(size_t workSize,     // size of work item: i.e. size of globalItemSize[] array.
    size_t* globalItemSize,
    size_t* localItemSize)// no idea how it works, just set to NULL for now.
{
    cl_int error = CL_SUCCESS;
    error = clEnqueueNDRangeKernel(m_cmdQueue, m_kernel, (cl_uint)workSize,
        NULL,   // offset always start from beginning
        globalItemSize, localItemSize, 0, NULL, NULL);
    checkCLError(error, "Enqueue NDRange Kernel Failed");
}

std::string CLSimpleWrapper::GetPlatformName(cl_platform_id id)
{
    size_t size = 0;
    clGetPlatformInfo(id, CL_PLATFORM_NAME, 0, nullptr, &size);

    std::string result;
    result.resize(size);
    clGetPlatformInfo(id, CL_PLATFORM_NAME, size,
        const_cast<char*> (result.data()), nullptr);

    return result;
}

std::string CLSimpleWrapper::GetDeviceName(cl_device_id id)
{
    size_t size = 0;
    clGetDeviceInfo(id, CL_DEVICE_NAME, 0, nullptr, &size);

    std::string result;
    result.resize(size);
    clGetDeviceInfo(id, CL_DEVICE_NAME, size,
        const_cast<char*> (result.data()), nullptr);

    return result;
}

void CLSimpleWrapper::clear()
{
    cl_int error = CL_SUCCESS;
    error = clFlush(m_cmdQueue);
    error = clFinish(m_cmdQueue);
    error = clReleaseKernel(m_kernel);
    error = clReleaseProgram(m_program);

    for ( cl_mem arg : m_args )
    {
        error = clReleaseMemObject(arg);
    }
    m_args.clear();

    error = clReleaseCommandQueue(m_cmdQueue);
    error = clReleaseContext(m_context);
}

void CLSimpleWrapper::checkCLError(cl_int error, std::string err_msg)
{
    if ( error != CL_SUCCESS )
    {
        if ( "" == err_msg )
        {
            // generic error message
            std::cerr << "OpenCL call failed with error " << error << std::endl;
        }
        else
        {
            std::cerr << err_msg << " :" << error << std::endl;
        }
        std::exit(1);
    }
}

