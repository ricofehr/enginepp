/**
 *  @file serial_collision_engine.cc
 *  @brief serial version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/serial_collision_engine.h"

#include <tbb/tbb.h>

#include <iostream>

#include "nextfloor/universe/model3d.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

SerialCollisionEngine::SerialCollisionEngine()
{
    InitCollisionEngine();
}

void SerialCollisionEngine::InitCollisionEngine() {
    using nextfloor::core::CommonServices;
    granularity_ = CommonServices::getConfig()->getCollisionGranularity();
}

float SerialCollisionEngine::ComputeCollision(nextfloor::universe::Model3D* target,
                                              nextfloor::universe::Model3D* obstacle)
{
    auto target_border = target->border();
    auto obstacle_border = obstacle->border();

    for (float fact = 1.0f; fact <= granularity_; fact += 1.0f) {
        float parted_move = fact / granularity_;
        if (target_border->IsObstacleInCollisionAfterPartedMove(obstacle_border, parted_move)) {
            return (fact - 1.0f) / granularity_;
        }
    }

    return 1.0f;
}

} // namespace helpers

} // namespace nextfloor
