/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/nearer_collision_engine.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

void NearerCollisionEngine::DetectCollision(nextfloor::objects::Mesh* target, nextfloor::objects::Mesh* obstacle)
{
    if (!target->IsCamera() && target->IsLastObstacle(obstacle)) {
        return;
    }

    auto distance = ComputeCollision(target, obstacle);
    target->UpdateObstacleIfNearer(obstacle, distance);
}


}  // namespace physics

}  // namespace nextfloor
