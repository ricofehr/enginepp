/**
 *  @file physic_factory.h
 *  @brief Abstract Factory Class for Physics concept
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_PHYSICFACTORY_H_
#define NEXTFLOOR_OBJECTS_PHYSICFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/border.h"
#include "nextfloor/objects/collision_engine.h"

namespace nextfloor {

namespace objects {

/**
 *  @class PhysicFactory
 *  @brief Abstract Factory Pattern for Physics concept
 */
class PhysicFactory {

public:
    virtual ~PhysicFactory() = default;

    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, float scale) const = 0;
    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual CollisionEngine* MakeCollisionEngine() const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_PHYSICFACTORY_H_
