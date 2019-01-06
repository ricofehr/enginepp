/**
 *  @file tbb_collision_engine.h
 *  @brief TbbCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_TBBCOLLISIONNEXTFLOOR_H_
#define PHYSICS_TBBCOLLISIONNEXTFLOOR_H_

#include <tbb/mutex.h>

#include "nextfloor/physics/collision_engine.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::physics
 *  @brief Physics and collision engines
 */
namespace physics {

/**
 *  @class TbbCollisionEngine
 *  @brief Implements inteltbbplus algorithm for collision computes
 */
class TbbCollisionEngine : public CollisionEngine {

public:

    /**
     *  Default Move constructor
     */
    TbbCollisionEngine(TbbCollisionEngine&&) = default;

    /**
     *  Default Move assignment
     */
    TbbCollisionEngine& operator=(TbbCollisionEngine&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    TbbCollisionEngine(const TbbCollisionEngine&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    TbbCollisionEngine& operator=(const TbbCollisionEngine&) = delete;

    /**
     *  Default destructor
     */
    ~TbbCollisionEngine() override = default;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
     */
    inline static TbbCollisionEngine* Instance()
    {
        static bool sIsInit = false;
        /* Raw pointers because static vars */
        static auto sInstance = new TbbCollisionEngine;
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

    /**
     *  Compute collision distance between borders of 2 objects
     *  Thanks to tbbplus paralell processing
     *  @param box1 includes the coords for the first border and the moving vector
     *  @param box2 includes the coords for the second border and the moving vector
     *  @return distance between the 2 borders
     */
    float ComputeCollision(float box1[], float box2[]) override final;

private:

    /**
     *  Default Constructor
     *  Protected scope ensure sole instance
     */
    TbbCollisionEngine() = default;

    /**
     *  Init tbbplus parallell context
     */
    void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace nextfloor

#endif // PHYSICS_TBBCOLLISIONNEXTFLOOR_H_
