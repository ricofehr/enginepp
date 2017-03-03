/*
* Proxycl utility file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/helpers/proxycl.h"

#include <iostream>
#include <fstream>
#include <string>

namespace engine {
namespace helpers {

void ProxyCL::InitCollisionCL() {

    try {
        /* Query for platforms */
        std::vector <cl::Platform> platforms;
        cl::Platform::get(&platforms);

        /* Get a list of devices on this platform */
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        /* Create a context for the devices */
        cl::Context context(devices);

        /* Create a command-queue for the first device */
        cl_queue = cl::CommandQueue(context, devices[0]);

        /* Create the memory buffers */
        bufferin.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float)));
        bufferin.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float)));
        bufferout.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY, 10 * sizeof(float)));

        /* Read the program Source */
        std::ifstream source_file("cl/collision_kernel.cl");
        std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, source_code);

        /* Create the program from the source code */
        cl::Program program = cl::Program(context, source);

        /* Compile the program for the devices */
        program.build(devices);

        /* Create the kernel */
        cl_kernel = cl::Kernel(program, "collision");
    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(1);
    }
 }

/* Init cl collision kernel */
std::vector<float> ProxyCL::ComputeCollisionCL(float box1[], float box2[])
{
    float distances[10] {1.0f};

    /* Copy the input data to the input buffer */
    cl_queue.enqueueWriteBuffer(bufferin[0], CL_TRUE, 0, 9 * sizeof(float), box1);
    cl_queue.enqueueWriteBuffer(bufferin[1], CL_TRUE, 0, 9 * sizeof(float), box2);

    /* Set kernel arguments */
    cl_kernel.setArg(0, bufferin[0]);
    cl_kernel.setArg(1, bufferin[1]);
    cl_kernel.setArg(2, bufferout[0]);

    /* Execute the kernel (1 workitem in 10 workgroup => compute distance for the 10 fact) */
    cl::NDRange global(10);
    cl::NDRange local(1);
    cl_queue.enqueueNDRangeKernel(cl_kernel, cl::NullRange, global, local);

    /* Copy the output data back to the host */
    cl_queue.enqueueReadBuffer(bufferout[0], CL_TRUE, 0, 10 * sizeof(float), distances);

    return std::vector<float>(std::begin(distances), std::end(distances));
}

}//namespace helpers
}//namespace engine
