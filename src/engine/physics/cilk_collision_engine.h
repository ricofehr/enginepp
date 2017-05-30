/*
 * CilkCollisionEngine class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Use of Singleton pattern because need a sole CollisionEngine for program.
 */

#ifndef PHYSICS_CILKCOLLISIONENGINE_H_
#define PHYSICS_CILKCOLLISIONENGINE_H_

#include <tbb/mutex.h>

#include "engine/physics/collision_engine.h"

namespace engine {
namespace physics {

class CilkCollisionEngine : public CollisionEngine {

public:

    ~CilkCollisionEngine() override = default;

    float ComputeCollision(float box1[], float box2[]) override final;

    inline static CilkCollisionEngine *Instance() {
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

    CilkCollisionEngine(){};
    CilkCollisionEngine(const CilkCollisionEngine&) = default;
    CilkCollisionEngine& operator=(const CilkCollisionEngine&) = default;

    void InitCollisionEngine() override final;
};

}//namespace parallell
}//namespace engine

#endif //PHYSICS_CILKCOLLISIONENGINE_H_
