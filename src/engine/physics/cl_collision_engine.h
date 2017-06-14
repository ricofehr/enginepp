/**
 *  CLCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *  Implements opencl algorithm for collision computes.
 */

#ifndef PHYSICS_CLCOLLISIONENGINE_H_
#define PHYSICS_CLCOLLISIONENGINE_H_

/* OpenCL 1.2 */
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#define CL_HPP_ENABLE_EXCEPTIONS

#include <vector>
#include <tbb/mutex.h>

#ifdef __APPLE__
    #include <OpenCL/cl2.hpp>
#else
    #include <CL/cl2.hpp>
#endif

#include "engine/physics/collision_engine.h"

namespace engine {

namespace physics {

class CLCollisionEngine : public CollisionEngine {

public:

    /**
     *  Default Move constructor
     */
    CLCollisionEngine(CLCollisionEngine&&) = default;

    /**
     *  Default Move assignment
     */
    CLCollisionEngine& operator=(CLCollisionEngine&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    CLCollisionEngine(const CLCollisionEngine&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    CLCollisionEngine& operator=(const CLCollisionEngine&) = delete;

    /**
     *  Default destructor
     */
    ~CLCollisionEngine() override = default;

    /**
     *  Compute collision distance between borders of 2 objects
     *  Thanks to opencl paralell processing
     *  @param box1 includes the coords for the first border and the moving vector
     *  @param box2 includes the coords for the second border and the moving vector
     *  @return distance between the 2 borders
     */
    float ComputeCollision(float box1[], float box2[]) override final;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
     */
    inline static CLCollisionEngine* Instance()
    {
        static bool sIsInit = false;
        /* Raw pointers because static vars */
        static auto sInstance = new CLCollisionEngine;
        static auto collision_mutex = new tbb::mutex;

        /*
         *  Init the context if not already done
         */
        collision_mutex->lock();
        if (!sIsInit) {
            sInstance->InitCollisionEngine();
            sIsInit = true;
        }
        collision_mutex->unlock();

        return sInstance;
    }

protected:

    /**
     *  Default Constructor
     *  Protected scope ensure sole instance
     */
    CLCollisionEngine() = default;

    /**
     *  Init opencl parallell context
     */
    void InitCollisionEngine() override final;


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

} // namespace engine

#endif // PHYSICS_CLCOLLISIONENGINE_H_
