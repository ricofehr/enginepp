/**
 *  @file mesh_physic_factory.h
 *  @brief MeshPhysicFactory Class for create Physic utility object
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_MESHPHYSICFACTORY_H_
#define NEXTFLOOR_PHYSIC_MESHPHYSICFACTORY_H_

#include "nextfloor/mesh/physic_factory.h"

namespace nextfloor {

namespace physic {

/**
 *  @class MeshPhysicFactory
 *  @brief Concrete factory for Physic concepts
 */
class MeshPhysicFactory : public nextfloor::mesh::PhysicFactory {

public:
    std::unique_ptr<nextfloor::mesh::Border> MakeBorder(const glm::vec3& location, float scale) const final;
    std::unique_ptr<nextfloor::mesh::Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::gameplay::CollisionEngine> MakeCollisionEngine() const final;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_MESHPHYSICFACTORY_H_
