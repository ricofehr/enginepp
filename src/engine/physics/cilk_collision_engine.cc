/*
 *   cilkplus version for CollisionEngine
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/physics/cilk_collision_engine.h"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_min.h>

#include "engine/core/config_engine.h"

namespace engine {

namespace physics {

void CilkCollisionEngine::InitCollisionEngine() {
    using engine::core::ConfigEngine;
    granularity_ = ConfigEngine::getSetting<int>("granularity");

    /* Define cpu cores number for parallell computes */
    if (ConfigEngine::getSetting<int>("workers_count")) {
        __cilkrts_set_param("nworkers", std::to_string(ConfigEngine::getSetting<int>("workers_count")).c_str());
    }
}

float CilkCollisionEngine::ComputeCollision(float box1[], float box2[])
{
    float x1, y1, z1, w1, h1, d1, move1x, move1y, move1z;
    float x2, y2, z2, w2, h2, d2, move2x, move2y, move2z;
    cilk::reducer_min<float> distance(1.0f);

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

    cilk_for (auto fact = 0; fact < granularity_; fact++) {
        x1 += move1x;
        y1 += move1y;
        z1 += move1z;
        x2 += move2x;
        y2 += move2y;
        z2 += move2z;

        if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
            y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                distance.calc_min(static_cast<float>(fact) / granularity_);
        }
    }

    return distance->get_value();
}

} // namespace helpers

} // namespace engine
