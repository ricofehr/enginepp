/**
 *  @file serial_collision.h
 *  @brief SerialCollision class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_SERIALCOLLISION_H_
#define NEXTFLOOR_PHYSICS_SERIALCOLLISION_H_

#include "nextfloor/physics/collision.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace physics {

/**
 *  @class SerialCollision
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialCollision : public Collision {

public:

    SerialCollision();

    SerialCollision(SerialCollision&&) = default;
    SerialCollision& operator=(SerialCollision&&) = default;
    SerialCollision(const SerialCollision&) = delete;
    SerialCollision& operator=(const SerialCollision&) = delete;

    virtual ~SerialCollision() override = default;

    virtual float ComputeCollision(nextfloor::objects::EngineObject* target,
                                   nextfloor::objects::EngineObject* obstacle) override final;

private:

    void InitCollisionEngine() override final;
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_SERIALCOLLISION_H_
