/**
 *  @file serial_collision_engine.cc
 *  @brief serial version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physic/serial_nearer_collision_engine.h"

#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace physic {

SerialNearerCollisionEngine::SerialNearerCollisionEngine(int granularity) : NearerCollisionEngine(granularity) {}

PartialMove SerialNearerCollisionEngine::ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle)
{
    PartialMove default_move{1.0f, glm::vec3(1.0f)};
    nextfloor::mesh::Border* target_border = target->border();
    nextfloor::mesh::Border* obstacle_border = obstacle->border();

    for (float factor = 1.0f; factor <= granularity_; factor += 1.0f) {
        float parted_move = factor / granularity_;
        if (target_border->IsObstacleInCollisionAfterPartedMove(*obstacle_border, parted_move)) {
            float factor_move = (factor - 1) / granularity_;
            return PartialMove{factor_move, glm::vec3(-1.0f)};
        }
    }

    return default_move;
}

}  // namespace physic

}  // namespace nextfloor
