/*
* OpenCL parallelisation class
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/parallell/cl_parallell.h"

#include <iostream>
#include <fstream>
#include <string>

namespace engine {
namespace parallell {

void CLParallell::InitCollisionParallell() {
    cl::Platform platform_target;
    cl::Device device_target;
    int max_cores = 0;
    size_t num;

    try {
        /* Query for platforms */
        std::vector <cl::Platform> platforms;
        cl::Platform::get(&platforms);

        /* Get a list of devices on this platform */
        std::vector<cl::Device> devices;
        for (auto &pf: platforms) {
            pf.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            for (auto &dev : devices) {
                size_t num;
                dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &num);
                if (num > max_cores) {
                    platform_target = pf;
                    device_target = dev;
                    max_cores = num;
                }
            }
        }

        std::cout << max_cores << std::endl;
        /* Ensure items size  are valid */
        //device_target.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &num);
        size_t nums[3]{0};
        device_target.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &nums);
        std::cout << nums[0] << "::" << granularity_ << std::endl;
        while (nums[0] < granularity_)
            granularity_ /= 2;

        /* Create a context for the devices */
        cl::Context context(device_target);

        /* Create a command-queue for the first device */
        cl_queue_ = cl::CommandQueue(context, device_target);

        /* Create the memory buffers */
        bufferin_.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float)));
        bufferin_.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, 9 * sizeof(float)));
        bufferout_.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY, granularity_ * sizeof(float)));

        /* Read the program Source */
        std::ifstream source_file("cl/collision_kernel.cl");
        std::string source_code(std::istreambuf_iterator<char>(source_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, source_code);

        /* Create the program from the source code */
        cl::Program program = cl::Program(context, source);

        /* Compile the program for the devices */
        program.build(std::vector<cl::Device>(1, device_target));

        /* Create the kernel */
        cl_kernel_ = cl::Kernel(program, "collision");

        /* Ensure items size and wkgroup size are valid */
        cl_kernel_.getWorkGroupInfo(device_target, CL_KERNEL_WORK_GROUP_SIZE, &num);
        std::cout << num << "::" << wk_size_ << std::endl;
        while (num < wk_size_)
            wk_size_ /= 2;
        std::cout << granularity_ << "::" << wk_size_ << std::endl;
    } catch(cl::Error error) {
        std::cout << error.what() << "(" << error.err() << ")" << std::endl;
        exit(1);
    }
 }

/* Init cl collision kernel */
float CLParallell::ComputeCollisionParallell(float box1[], float box2[])
{
    std::vector<float> distances(granularity_, 1.0f);
    float *distances_ptr = distances.data();

    /* Copy the input data to the input buffer */
    cl_queue_.enqueueWriteBuffer(bufferin_[0], CL_TRUE, 0, 9 * sizeof(float), box1);
    cl_queue_.enqueueWriteBuffer(bufferin_[1], CL_TRUE, 0, 9 * sizeof(float), box2);
    cl_queue_.enqueueWriteBuffer(bufferout_[0], CL_TRUE, 0, granularity_ * sizeof(float), distances_ptr);

    /* Set kernel arguments */
    cl_kernel_.setArg(0, bufferin_[0]);
    cl_kernel_.setArg(1, bufferin_[1]);
    cl_kernel_.setArg(2, bufferout_[0]);

    /* Execute the kernel (1 workitem in 10 workgroup => compute distance for the 10 fact) */
    cl::NDRange global(granularity_);
    cl::NDRange local(wk_size_);
    cl_queue_.enqueueNDRangeKernel(cl_kernel_, cl::NullRange, global, local);

    /* Copy the output data back to the host */
    cl_queue_.enqueueReadBuffer(bufferout_[0], CL_TRUE, 0, granularity_ * sizeof(float), distances_ptr);

    for (auto i = 0; i < granularity_; i++) {
        if (distances[i] != 1.0f)
            return distances[i];
    }

    return 1.0f;
}

}//namespace helpers
}//namespace engine
