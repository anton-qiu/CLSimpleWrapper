# CLSimpleWrapper
Author: Anton Qiu
Website: http://www.learnitthehardway.com
License: MIT

Simple C++ Wrapper for Open CL
I know there is official OpenCL C++ Binding for OpenCL, but this project is not to replace that. 

The goal of this wrapper is to simplify the process of writing OpenCL code in a few simple steps:
1. Write the OpenCL kernel code.
2. Create CLSimpleWrapper instance
3. Call CLSimpleWrapper::initOpenCL() with optional platform and device ID. If platform and device ID is not supplied, this function will select the first platform and device available.
4. Call CLSimpleWrapper::createCLKernel(), with the OpenCL kernel code and the program name.
5. Set the necessary arguments.
6. Call CLSimpleWrapper::executeKernel() to execute the Open CL code.
7. Read the execution result by calling CLSimpleWrapper::readBuffer()
8. Delete the CLSimpleWrapper instance.

Example of the usage is provided in the example directory.

