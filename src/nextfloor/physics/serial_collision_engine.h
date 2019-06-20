/**
 *  @file serial_collision_engine.h
 *  @brief SerialCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef PHYSICS_SERIALCOLLISIONNEXTFLOOR_H_
#define PHYSICS_SERIALCOLLISIONNEXTFLOOR_H_

#include <tbb/mutex.h>

#include "nextfloor/physics/collision_engine.h"

namespace nextfloor {

namespace physics {

/**
 *  @class SerialCollisionEngine
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialCollisionEngine : public CollisionEngine {

public:

    SerialCollisionEngine(SerialCollisionEngine&&) = default;

    SerialCollisionEngine& operator=(SerialCollisionEngine&&) = default;

    SerialCollisionEngine(const SerialCollisionEngine&) = delete;

    SerialCollisionEngine& operator=(const SerialCollisionEngine&) = delete;

    ~SerialCollisionEngine() override = default;

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole collision engine instance
     */
    inline static SerialCollisionEngine* Instance()
    {
        static auto sInstance = new SerialCollisionEngine;
        return sInstance;
    }

    virtual float ComputeCollision(nextfloor::universe::Model3D* target,
                                   nextfloor::universe::Model3D* obstacle) override final;

private:

    SerialCollisionEngine();

    void InitCollisionEngine() override final;
};

} // namespace parallell

} // namespace nextfloor

#endif // PHYSICS_SERIALCOLLISIONNEXTFLOOR_H_
