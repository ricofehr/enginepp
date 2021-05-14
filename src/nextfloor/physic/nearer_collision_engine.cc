/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physic/nearer_collision_engine.h"

namespace nextfloor {

namespace physic {

void NearerCollisionEngine::DetectCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle)
{
    /* 2 objects cannot themselves in collision, but player (camera) */
    if (!target->IsCamera() && target->IsLastObstacle(obstacle)) {
        return;
    }

    PartialMove collision_factor = ComputeCollision(target, obstacle);
    target->UpdateObstacleIfNearer(obstacle, collision_factor.distance_factor, collision_factor.movement_factor_update);
}

}  // namespace physic

}  // namespace nextfloor
