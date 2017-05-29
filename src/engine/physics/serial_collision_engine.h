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
    float ComputeCollision(float box1[], float box2[]) override final;

    inline static SerialCollisionEngine *Instance() {
        static auto sInstance = std::unique_ptr<SerialCollisionEngine>(new SerialCollisionEngine);
        static bool sIsInit = false;
        static tbb::mutex collision_mutex;

        /* Init the engine if not already done */
        collision_mutex.lock();
        if (!sIsInit) {
            sInstance->InitCollisionEngine();
            sIsInit = true;
        }
        collision_mutex.unlock();

        return sInstance.get();
    }

    ~SerialCollisionEngine() override = default;

protected:
    void InitCollisionEngine() override final;

    SerialCollisionEngine(){};
    SerialCollisionEngine(const SerialCollisionEngine&) = default;
    SerialCollisionEngine& operator=(const SerialCollisionEngine&) = default;

};

}//namespace parallell
}//namespace engine

#endif //PHYSICS_SERIALCOLLISIONENGINE_H_
