/**
 *  @file tbb_collision_engine.cc
 *  @brief Tbb version for CollisionEngine
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/physics/tbb_collision_engine.h"

#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace physics {

void TbbCollisionEngine::InitCollisionEngine() {
    using nextfloor::core::CommonServices;
    granularity_ = CommonServices::getConfig().getSetting<int>("granularity");
}

float TbbCollisionEngine::ComputeCollision(float box1[], float box2[])
{
    float x1, y1, z1, w1, h1, d1, move1x, move1y, move1z;
    float x2, y2, z2, w2, h2, d2, move2x, move2y, move2z;
    float distance(1.0f);

    x1 = box1[0];
    y1 = box1[1];
    z1 = box1[2];
    w1 = box1[3];
    h1 = box1[4];
    d1 = box1[5];
    move1x = box1[6] / granularity_;
    move1y = box1[7] / granularity_;
    move1z = box1[8] / granularity_;
    x2 = box2[0];
    y2 = box2[1];
    z2 = box2[2];
    w2 = box2[3];
    h2 = box2[4];
    d2 = box2[5];
    move2x = box2[6] / granularity_;
    move2y = box2[7] / granularity_;
    move2z = box2[8] / granularity_;

    tbb::parallel_for (1, (granularity_+1), 1, [&](int fact) {
        auto test_x1 = x1 + fact * move1x;
        auto test_y1 = y1 + fact * move1y;
        auto test_z1 = z1 + fact * move1z;
        auto test_x2 = x2 + fact * move2x;
        auto test_y2 = y2 + fact * move2y;
        auto test_z2 = z2 + fact * move2z;

        if (test_x2 <= test_x1 + w1 && test_x2 + w2 >= test_x1 &&
            test_y2 + h2 <= test_y1 && test_y2 >= test_y1 + h1 &&
            test_z2 >= test_z1 + d1 && test_z2 + d2 <= test_z1) {
            auto distcol = static_cast<float>(fact-1) / granularity_;

            /* Raw pointers because static vars */
            static auto distance_mutex = new tbb::mutex;

            tbb::mutex::scoped_lock lock_map(*distance_mutex);
            if (distcol < distance) {
                distance = distcol;
            }
        }
    });

    return distance;
}

} // namespace helpers

} // namespace nextfloor
