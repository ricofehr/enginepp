/*
 * SerialCollisionEngine class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Use of Singleton pattern because need a sole CollisionEngine for program.
 */

#ifndef PHYSICS_SERIALCOLLISIONENGINE_H_
#define PHYSICS_SERIALCOLLISIONENGINE_H_

#include <tbb/mutex.h>

#include "engine/physics/collision_engine.h"

namespace engine {
namespace physics {

class SerialCollisionEngine : public CollisionEngine {

public:
    
    ~SerialCollisionEngine() override = default;

    float ComputeCollision(float box1[], float box2[]) override final;

    inline static SerialCollisionEngine *Instance() {
        static bool sIsInit = false;
        /* Raw pointers because static vars */
        static auto sInstance = new SerialCollisionEngine;
        static auto collision_mutex = new tbb::mutex;

        /* Init the engine if not already done */
        collision_mutex->lock();
        if (!sIsInit) {
            sInstance->InitCollisionEngine();
            sIsInit = true;
        }
        collision_mutex->unlock();

        return sInstance;
    }


protected:

    SerialCollisionEngine(){};
    SerialCollisionEngine(const SerialCollisionEngine&) = default;
    SerialCollisionEngine& operator=(const SerialCollisionEngine&) = default;

    void InitCollisionEngine() override final;
};

}//namespace parallell
}//namespace engine

#endif //PHYSICS_SERIALCOLLISIONENGINE_H_
