
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <string>
#include <stdlib.h> 
#include <cmath>

#include "CLSimpleWrapper.h"


int main(int argc, char* argv[])
{
    CLSimpleWrapper cl_wrapper;

    cl_wrapper.initOpenCL(-1, -1, true);   // print the CL Info without creating initializing.
    return 0;
}
