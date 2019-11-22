/**
 *  @file serial_nearer_collision_engine.h
 *  @brief SerialNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_

#include "nextfloor/physics/nearer_collision_engine.h"

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace physics {

/**
 *  @class SerialNearerCollisionEngine
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialNearerCollisionEngine : public NearerCollisionEngine {

public:
    SerialNearerCollisionEngine();
    ~SerialNearerCollisionEngine() final = default;

    float ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle) final;

private:
    void InitCollisionEngine() final;
};

}  // namespace physics

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSICS_SERIALNEARERCOLLISIONENGINE_H_
