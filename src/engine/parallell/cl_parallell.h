#ifndef ENGINE_PARALLELL_CLPARALLELL_H_
#define ENGINE_PARALLELL_CLPARALLELL_H_

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

#include "engine/parallell/engine_parallell.h"

namespace engine {
namespace parallell {

class CLParallell : public EngineParallell {

public:
    void InitCollisionParallell() override final;
    float ComputeCollisionParallell(float box1[], float box2[]) override final;
    ~CLParallell() override = default;

private:
    cl::Kernel cl_kernel_;
    cl::CommandQueue cl_queue_;
    std::vector<cl::Buffer> bufferin_;
    std::vector<cl::Buffer> bufferout_;
    int granularity_{1024};
    int wk_size_{128};
};

}//namespace parallell
}//namespace engine

#endif //ENGINE_PARALLELL_CLPARALLELL_H_
