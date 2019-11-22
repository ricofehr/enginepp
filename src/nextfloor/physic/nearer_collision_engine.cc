/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physic/nearer_collision_engine.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physic {

void NearerCollisionEngine::DetectCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle)
{
    if (!target->IsCamera() && target->IsLastObstacle(obstacle)) {
        return;
    }

    auto distance = ComputeCollision(target, obstacle);
    target->UpdateObstacleIfNearer(obstacle, distance);
}

}  // namespace physic

}  // namespace nextfloor
