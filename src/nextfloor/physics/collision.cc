/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/collision.h"

#include <iostream>
#include <vector>

#include "nextfloor/objects/engine_object.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

void Collision::DetectCollision(nextfloor::objects::EngineObject* target,
                                nextfloor::objects::EngineObject* obstacle)
{
    if (target->IsLastObstacle(obstacle)) {
        return;
    }

    auto distance = ComputeCollision(target, obstacle);
    target->UpdateObstacleIfNearer(obstacle, distance);
}


} // namespace physics

} // namespace nextfloor
