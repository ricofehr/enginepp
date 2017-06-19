/**
 *  @file serial_collision_engine.h
 *  @brief SerialCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_SERIALCOLLISIONENGINE_H_
#define PHYSICS_SERIALCOLLISIONENGINE_H_

#include <tbb/mutex.h>

#include "engine/physics/collision_engine.h"

/**
 *  @namespace engine
 *  @brief Common parent namespace
 */
namespace engine {

/**
 *  @namespace engine::physics
 *  @brief Physics and collision engines
 */
namespace physics {

/**
 *  @class SerialCollisionEngine
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialCollisionEngine : public CollisionEngine {

public:

    /**
     *  Default Move constructor
     */
    SerialCollisionEngine(SerialCollisionEngine&&) = default;

    /**
     *  Default Move assignment
     */
    SerialCollisionEngine& operator=(SerialCollisionEngine&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    SerialCollisionEngine(const SerialCollisionEngine&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    SerialCollisionEngine& operator=(const SerialCollisionEngine&) = delete;

    /**
     *  Default destructor
     */
    ~SerialCollisionEngine() override = default;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
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

    /**
     *  Compute collision distance between borders of 2 objects
     *  Thanks to a serial compute in only one cpu core.
     *  @param box1 includes the coords for the first border and the moving vector
     *  @param box2 includes the coords for the second border and the moving vector
     *  @return distance between the 2 borders
     */
    float ComputeCollision(float box1[], float box2[]) override final;

private:

    /*
     *  Default Constructor
     *  Private scope ensure sole instance
     */
    SerialCollisionEngine() = default;

    /**
     *  Init serial context
     */
    void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace engine

#endif // PHYSICS_SERIALCOLLISIONENGINE_H_
