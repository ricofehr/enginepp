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

    float ComputeCollision(float box1[], float box2[]) override final;

    inline static CilkCollisionEngine *Instance() {
        static auto sInstance = std::unique_ptr<CilkCollisionEngine>(new CilkCollisionEngine);
        static bool sIsInit = false;
        static tbb::mutex collision_mutex;

        /* 
         *  Init the engine if not already done
         *  TODO: Protect from parallell access
         */
        collision_mutex.lock();
        if (!sIsInit) {
            sInstance->InitCollisionEngine();
            sIsInit = true;
        }
        collision_mutex.unlock();

        return sInstance.get();
    }

    ~CilkCollisionEngine() override = default;

protected:
    void InitCollisionEngine() override final;

    CilkCollisionEngine(){};
    CilkCollisionEngine(const CilkCollisionEngine&) = default;
    CilkCollisionEngine& operator=(const CilkCollisionEngine&) = default;
};

}//namespace parallell
}//namespace engine

#endif //PHYSICS_CILKCOLLISIONENGINE_H_
