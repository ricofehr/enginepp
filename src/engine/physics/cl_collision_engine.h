/*
 * CLCollisionEngine class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Use of Singleton pattern because need a sole CollisionEngine for program.
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

    ~CLCollisionEngine() override = default;

    float ComputeCollision(float box1[], float box2[]) override final;

    inline static CLCollisionEngine *Instance() {
        static bool sIsInit = false;
        /* Raw pointers because static vars */
        static auto sInstance = new CLCollisionEngine;
        static auto collision_mutex = new tbb::mutex;

        /*
         *  Init the engine if not already done
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

    CLCollisionEngine(){};
    CLCollisionEngine(const CLCollisionEngine&) = default;
    CLCollisionEngine& operator=(const CLCollisionEngine&) = default;

    void InitCollisionEngine() override final;

    cl::Kernel cl_kernel_;
    cl::CommandQueue cl_queue_;
    std::vector<cl::Buffer> bufferin_;
    std::vector<cl::Buffer> bufferout_;
    tbb::mutex collision_mutex_;
    int wk_size_{32};
};

}//namespace parallell
}//namespace engine

#endif //PHYSICS_CLCOLLISIONENGINE_H_
