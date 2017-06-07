/*
 *  CilkCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Implements intelcilkplus algorithm for collision computes.
 */

#ifndef PHYSICS_CILKCOLLISIONENGINE_H_
#define PHYSICS_CILKCOLLISIONENGINE_H_

#include <tbb/mutex.h>

#include "engine/physics/collision_engine.h"

namespace engine {

namespace physics {

class CilkCollisionEngine : public CollisionEngine {

public:

    /*
     *  Default destructor
     */
    ~CilkCollisionEngine() override = default;

    /*
     *  Compute collision distance between 2 borders (box1 and box2)
     *  Thanks to cilkplus paralell processing
     */
    float ComputeCollision(float box1[], float box2[]) override final;

    /*
     *  Return (and allocates if needed) sole Instance
     */
    inline static CilkCollisionEngine* Instance()
    {
        static bool sIsInit = false;
        /* Raw pointers because static vars */
        static auto sInstance = new CilkCollisionEngine;
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

    /*
     *  Constructor
     *  Protected scope ensure sole instance
     */
    CilkCollisionEngine(){};
    CilkCollisionEngine(const CilkCollisionEngine&) = default;
    CilkCollisionEngine& operator=(const CilkCollisionEngine&) = default;

    /*
     *  Init cilkplus parallell context
     */
    void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace engine

#endif // PHYSICS_CILKCOLLISIONENGINE_H_
