/**
 *  @file tbb_collision.cc
 *  @brief Tbb version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/tbb_collision.h"

#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

TbbCollision::TbbCollision()
{
    InitCollisionEngine();
}

void TbbCollision::InitCollisionEngine()
{
    using nextfloor::core::CommonServices;
    granularity_ = CommonServices::getConfig()->getCollisionGranularity();
}

float TbbCollision::ComputeCollision(nextfloor::objects::EngineObject* target,
                                     nextfloor::objects::EngineObject* obstacle)
{
    float distance(1.0f);
    auto target_border = target->border();
    auto obstacle_border = obstacle->border();
    tbb::mutex distance_mutex;

    tbb::parallel_for (1, (granularity_+1), 1, [&](int fact) {
        float parted_move = static_cast<float>(fact) / granularity_;
        if (target_border->IsObstacleInCollisionAfterPartedMove(obstacle_border, parted_move)) {
            float test_collision = static_cast<float>(fact-1) / granularity_;

            tbb::mutex::scoped_lock lock_map(distance_mutex);
            if (test_collision < distance) {
                distance = test_collision;
            }
        }
    });

    return distance;
}

} // namespace physics

} // namespace nextfloor
