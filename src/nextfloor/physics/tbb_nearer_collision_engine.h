/**
 *  @file tbb_nearer_collision_engine.h
 *  @brief TbbNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_TBBNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSICS_TBBNEARERCOLLISIONENGINE_H_

#include "nextfloor/physics/nearer_collision_engine.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace physics {

/**
 *  @class TbbNearerCollisionEngine
 *  @brief Implements inteltbbplus algorithm for collision computes
 */
class TbbNearerCollisionEngine : public NearerCollisionEngine {

public:
    TbbNearerCollisionEngine();
    ~TbbNearerCollisionEngine() final = default;

    float ComputeCollision(nextfloor::objects::Mesh* target, nextfloor::objects::Mesh* obstacle) final;

private:
    void InitCollisionEngine() final;
};

}  // namespace physics

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSICS_TBBNEARERCOLLISIONENGINE_H_
