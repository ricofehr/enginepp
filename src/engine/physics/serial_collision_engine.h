/*
 *  SerialCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  Implements Serial algorithm for collision computes.
 */

#ifndef PHYSICS_SERIALCOLLISIONENGINE_H_
#define PHYSICS_SERIALCOLLISIONENGINE_H_

#include <tbb/mutex.h>

#include "engine/physics/collision_engine.h"

namespace engine {

namespace physics {

class SerialCollisionEngine : public CollisionEngine {

public:

    /*
     *  Default destructor
     */
    ~SerialCollisionEngine() override = default;

    /*
     *  Compute collision distance between 2 borders (box1 and box2)
     *  Thanks to a serial compute in only one cpu core.
     */
    float ComputeCollision(float box1[], float box2[]) override final;

    /*
     *  Return (and allocates if needed) sole Instance
     */
    inline static SerialCollisionEngine* Instance()
    {
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

    /*
     *  Constructor
     *  Protected scope ensure sole instance
     */
    SerialCollisionEngine(){};
    SerialCollisionEngine(const SerialCollisionEngine&) = default;
    SerialCollisionEngine& operator=(const SerialCollisionEngine&) = default;

    /*
     *  Init serial context
     */
    void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace engine

#endif // PHYSICS_SERIALCOLLISIONENGINE_H_
