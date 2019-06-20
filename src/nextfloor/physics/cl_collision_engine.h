/**
 *  @file cl_collision_engine.h
 *  @brief CLCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_CLCOLLISIONNEXTFLOOR_H_
#define PHYSICS_CLCOLLISIONNEXTFLOOR_H_

/* OpenCL 1.2 */
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <vector>
#include <tbb/mutex.h>

#include <GL/glew.h>
#ifdef __APPLE__
    #include <OpenCL/cl2.hpp>
#else
    #include <CL/cl2.hpp>
#endif

#include "nextfloor/physics/collision_engine.h"

namespace nextfloor {

namespace physics {

/**
 *  @class CLCollisionEngine
 *  @brief Implements opencl algorithm for collision computes
 */
class CLCollisionEngine : public CollisionEngine {

public:

    CLCollisionEngine(CLCollisionEngine&&) = default;

    CLCollisionEngine& operator=(CLCollisionEngine&&) = default;

    CLCollisionEngine(const CLCollisionEngine&) = delete;

    CLCollisionEngine& operator=(const CLCollisionEngine&) = delete;

    ~CLCollisionEngine() override = default;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
     */
    inline static CLCollisionEngine* Instance()
    {
        static auto sInstance = new CLCollisionEngine;
        return sInstance;
    }

    virtual float ComputeCollision(nextfloor::universe::Model3D* target,
                                   nextfloor::universe::Model3D* obstacle) override final;

protected:

    CLCollisionEngine();

    virtual void InitCollisionEngine() override final;

    /** Opencl Kernel Object */
    cl::Kernel cl_kernel_;

    /** Opencl Execution Queue  Object */
    cl::CommandQueue cl_queue_;

    /** Opencl Input Buffer */
    std::vector<cl::Buffer> bufferin_;

    /** Opencl Output Buffer */
    std::vector<cl::Buffer> bufferout_;

    /** Ensures thread safe execution */
    tbb::mutex collision_mutex_;

    /** Opencl Workgroup size */
    int wk_size_{32};
};

} // namespace parallell

} // namespace nextfloor

#endif // PHYSICS_CLCOLLISIONNEXTFLOOR_H_
