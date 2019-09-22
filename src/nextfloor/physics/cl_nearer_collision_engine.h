/**
 *  @file cl_nearer_collision_engine.h
 *  @brief ClNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_CLNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSICS_CLNEARERCOLLISIONENGINE_H_

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

#include "nextfloor/physics/nearer_collision_engine.h"

namespace nextfloor {

namespace physics {

/**
 *  @class ClNearerCollisionEngine
 *  @brief Implements opencl algorithm for collision computes
 */
class ClNearerCollisionEngine : public NearerCollisionEngine {

public:
    ClNearerCollisionEngine();
    ~ClNearerCollisionEngine() final = default;

    ClNearerCollisionEngine(ClNearerCollisionEngine&&) = delete;
    ClNearerCollisionEngine& operator=(ClNearerCollisionEngine&&) = delete;
    ClNearerCollisionEngine(const ClNearerCollisionEngine&) = delete;
    ClNearerCollisionEngine& operator=(const ClNearerCollisionEngine&) = delete;

    float ComputeCollision(nextfloor::objects::Mesh* target, nextfloor::objects::Mesh* obstacle) final;

protected:
    void InitCollisionEngine() final;

private:
    void HandleErrorOnInit(cl::Error error);

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

}  // namespace physics

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSICS_CLNEARERCOLLISIONENGINE_H_
