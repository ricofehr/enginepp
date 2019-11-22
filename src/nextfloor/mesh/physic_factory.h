/**
 *  @file physic_factory.h
 *  @brief Abstract Factory Class for Physics concept
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_PHYSICFACTORY_H_
#define NEXTFLOOR_MESH_PHYSICFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/mesh/border.h"
#include "nextfloor/gameplay/collision_engine.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class PhysicFactory
 *  @brief Abstract Factory Pattern for Physics concept
 */
class PhysicFactory {

public:
    virtual ~PhysicFactory() = default;

    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, float scale) const = 0;
    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::CollisionEngine> MakeCollisionEngine() const = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_PHYSICFACTORY_H_
