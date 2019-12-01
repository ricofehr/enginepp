/**
 *  @file tbb_nearer_collision_engine.h
 *  @brief TbbNearerCollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_TBBNEARERCOLLISIONENGINE_H_
#define NEXTFLOOR_PHYSIC_TBBNEARERCOLLISIONENGINE_H_

#include "nextfloor/physic/nearer_collision_engine.h"

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace physic {

/**
 *  @class TbbNearerCollisionEngine
 *  @brief Implements inteltbbplus algorithm for collision computes
 */
class TbbNearerCollisionEngine : public NearerCollisionEngine {

public:
    TbbNearerCollisionEngine(int granularity);
    ~TbbNearerCollisionEngine() final = default;

    float ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle) final;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_TBBNEARERCOLLISIONENGINE_H_
