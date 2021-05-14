/**
 *  @file collision_engine.h
 *  @brief CollisionEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_COLLISIONENGINE_H_
#define NEXTFLOOR_GAMEPLAY_COLLISIONENGINE_H_

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace physic {

/**
 *  Define partial move with factor of movement and new movement axes
 */
typedef struct {
    float distance_factor;
    glm::vec3 movement_factor_update;
} PartialMove;

/**
 *  @class EngineCollision
 *  @brief Interface who manage collisition computes between 3d models\n
 */
class CollisionEngine {

public:
    virtual ~CollisionEngine() = default;

    /* Template Method : Detect if a collision exists between target and obstacle. */
    virtual void DetectCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle) = 0;

    /**
     *  Primitive Operation subclassed: compute collision distance between borders of 2 objects
     *  @return parted (as fraction of setted move) distance between the 2 borders
     */
    virtual PartialMove ComputeCollision(nextfloor::mesh::Mesh* target, nextfloor::mesh::Mesh* obstacle) = 0;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_COLLISIONENGINE_H_
