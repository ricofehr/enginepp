/**
 *  @file serial_nearer_collision_engine.h
 *  @brief SerialNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_SERIALNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSIC_SERIALNEARERCOLLISIONENGINE_H_

#include "nextfloor/physic/nearer_collision_engine.h"

namespace nextfloor {

namespace physic {

/**
 *  @class SerialNearerCollisionEngine
 *  @brief Implements Serial algorithm for collision computes
 */
class SerialNearerCollisionEngine : public NearerCollisionEngine {

public:
    SerialNearerCollisionEngine(int granularity);
    ~SerialNearerCollisionEngine() final = default;

    float ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle) final;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_SERIALNEARERCOLLISIONENGINE_H_
