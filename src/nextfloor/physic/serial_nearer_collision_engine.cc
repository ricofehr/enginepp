/**
 *  @file serial_collision_engine.cc
 *  @brief serial version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physic/serial_nearer_collision_engine.h"

#include <tbb/tbb.h>

namespace nextfloor {

namespace physic {

SerialNearerCollisionEngine::SerialNearerCollisionEngine(int granularity) : NearerCollisionEngine(granularity) {}

float SerialNearerCollisionEngine::ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle)
{
    auto target_border = target->border();
    auto obstacle_border = obstacle->border();

    for (float fact = 1.0f; fact <= granularity_; fact += 1.0f) {
        float parted_move = fact / granularity_;
        if (target_border->IsObstacleInCollisionAfterPartedMove(*obstacle_border, parted_move)) {
            return (fact - 1.0f) / granularity_;
        }
    }

    return 1.0f;
}

}  // namespace physic

}  // namespace nextfloor
