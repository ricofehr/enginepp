/**
 *  @file cl_collision_engine.cc
 *  @brief CLCollisionEngine class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/cl_collision_engine.h"

#include <iostream>
#include <fstream>

#include "nextfloor/core/config_engine.h"

namespace nextfloor {

namespace physics {

void CLCollisionEngine::InitCollisionEngine()
{
    cl::Platform platform_target;
    cl::Device device_target;
    int max_cores = 0;
    size_t num;

    using nextfloor::core::ConfigEngine;
    granularity_ = ConfigEngine::getSetting<int>("granularity");

    try {
        /* Query for platforms */
        std::vector <cl::Platform> platforms;
        cl::Platform::get(&platforms);

        /* Select best devices in the workstation */
        std::vector<cl::Device> devices;
        for (auto &pf: platforms) {
            pf.getDevices(CL_DEVICE_TYPE_ALL, &devices);
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

        /* Ensure items size  are valid */
        size_t nums[3]{0};
        device_target.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &nums);
        while (nums[0] < granularity_) {
            granularity_ /= 2;
        }

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
        while (num < wk_size_) {
            wk_size_ /= 2;
        }
    } catch(cl::Error error) {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        exit(1);
    }
}

float CLCollisionEngine::ComputeCollision(float box1[], float box2[])
{
    /* All tbb threads share same opencl objects, need mutex */
    collision_mutex_.lock();

    float* distances_ptr = new float[granularity_];
    float ret = 1.0f;

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
        if (distances_ptr[i] != 1.0f) {
            ret = distances_ptr[i];
            break;
        }
    }

    delete distances_ptr;
    collision_mutex_.unlock();

    return ret;
}

} // namespace helpers

} // namespace nextfloor
