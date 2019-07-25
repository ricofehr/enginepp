/**
 *  @file cl_collision_engine.h
 *  @brief CLCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_CLCOLLISION_H_
#define NEXTFLOOR_PHYSICS_CLCOLLISION_H_

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

#include "nextfloor/physics/collision.h"

namespace nextfloor {

namespace physics {

/**
 *  @class ClCollision
 *  @brief Implements opencl algorithm for collision computes
 */
class ClCollision : public Collision {

public:

    ClCollision();

    ClCollision(ClCollision&&) = default;
    ClCollision& operator=(ClCollision&&) = default;
    ClCollision(const ClCollision&) = delete;
    ClCollision& operator=(const ClCollision&) = delete;

    virtual ~ClCollision() override = default;

    virtual float ComputeCollision(nextfloor::objects::EngineObject* target,
                                   nextfloor::objects::EngineObject* obstacle) override final;

protected:

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

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_CLCOLLISION_H_
