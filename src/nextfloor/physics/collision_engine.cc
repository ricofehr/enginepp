/**
 *  @file collision_engine.cc
 *  @brief CollisionEngine class file.
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/collision_engine.h"

#include <iostream>
#include <vector>

#include "nextfloor/universe/model3d.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

void CollisionEngine::DetectCollision(nextfloor::universe::Model3D* target,
                                      nextfloor::universe::Model3D* obstacle)
{
    if (target->IsLastObstacle(obstacle)) {
        return;
    }

    auto distance = ComputeCollision(target, obstacle);
    target->UpdateObstacleIfNearer(obstacle, distance);
}



} // namespace helpers

} // namespace nextfloor
