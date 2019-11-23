/**
 *  @file collision_engine_factory.h
 *  @brief Abstract Factory Class for CollisionEngines
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_COLLISIONENGINEFACTORY_H_
#define NEXTFLOOR_MESH_COLLISIONENGINEFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/physic/collision_engine.h"

namespace nextfloor {

namespace physic {

/**
 *  @class CollisionEngineFactory
 *  @brief Abstract Factory Pattern for Collision Engines
 */
class CollisionEngineFactory {

public:
    virtual ~CollisionEngineFactory() = default;
    virtual std::unique_ptr<CollisionEngine> MakeCollisionEngine() const = 0;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_COLLISIONENGINEFACTORY_H_
