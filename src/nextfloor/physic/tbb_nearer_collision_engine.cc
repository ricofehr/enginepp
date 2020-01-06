/**
 *  @file tbb_collision.cc
 *  @brief Tbb version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physic/tbb_nearer_collision_engine.h"

#include <tbb/tbb.h>

#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace physic {

TbbNearerCollisionEngine::TbbNearerCollisionEngine(int granularity) : NearerCollisionEngine(granularity) {}

float TbbNearerCollisionEngine::ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle)
{
    float distance(1.0f);
    nextfloor::mesh::Border* target_border = target->border();
    nextfloor::mesh::Border* obstacle_border = obstacle->border();
    tbb::mutex distance_mutex;

    tbb::parallel_for(1, (granularity_ + 1), 1, [&](int factor) {
        float parted_move = static_cast<float>(factor) / granularity_;
        if (target_border->IsObstacleInCollisionAfterPartedMove(*obstacle_border, parted_move)) {
            float test_collision = static_cast<float>(factor - 1) / granularity_;

            tbb::mutex::scoped_lock lock_map(distance_mutex);
            if (test_collision < distance) {
                distance = test_collision;
            }
        }
    });

    return distance;
}

}  // namespace physic

}  // namespace nextfloor
