#ifndef ENGINE_HELPERS_PROXYCL_H_
#define ENGINE_HELPERS_PROXYCL_H_

/* OpenCL 1.2 */
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <vector>

#ifdef __APPLE__
    #include <OpenCL/cl2.hpp>
#else
    #include <CL/cl2.hpp>
#endif

namespace engine {
namespace helpers {

class ProxyCL {

public:
    void InitCollisionCL();
    std::vector<float> ComputeCollisionCL(float box1[], float box2[]);

private:
    cl::Kernel cl_kernel;
    cl::CommandQueue cl_queue;
    std::vector<cl::Buffer> bufferin;
    std::vector<cl::Buffer> bufferout;

};

}//namespace helpers
}//namespace engine

#endif //ENGINE_HELPERS_PROXYCL_H_
