/**
 *  @file serial_nearer_collision_engine.h
 *  @brief SerialNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_

#include "nextfloor/physics/nearer_collision_engine.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace physics {

/**
 *  @class SerialNearerCollisionEngine
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialNearerCollisionEngine : public NearerCollisionEngine {

public:

    SerialNearerCollisionEngine();

    SerialNearerCollisionEngine(SerialNearerCollisionEngine&&) = default;
    SerialNearerCollisionEngine& operator=(SerialNearerCollisionEngine&&) = default;
    SerialNearerCollisionEngine(const SerialNearerCollisionEngine&) = delete;
    SerialNearerCollisionEngine& operator=(const SerialNearerCollisionEngine&) = delete;

    virtual ~SerialNearerCollisionEngine() override = default;

    virtual float ComputeCollision(nextfloor::objects::Mesh* target,
                                   nextfloor::objects::Mesh* obstacle) override final;

private:

    void InitCollisionEngine() override final;
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_
