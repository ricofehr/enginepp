/**
 *  @file mesh_physic_factory.h
 *  @brief MeshPhysicFactory Class for create Physic utility object
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSICS_MESHPHYSICFACTORY_H_
#define NEXTFLOOR_PHYSICS_MESHPHYSICFACTORY_H_

#include "nextfloor/objects/physic_factory.h"

#include "nextfloor/polygons/polygon_factory.h"

namespace nextfloor {

namespace physics {

/**
 *  @class MeshPhysicFactory
 *  @brief Concrete factory for Physic concepts
 */
class MeshPhysicFactory : public nextfloor::objects::PhysicFactory {

public:
    MeshPhysicFactory(nextfloor::polygons::PolygonFactory* polygon_factory);
    ~MeshPhysicFactory() final = default;

    std::unique_ptr<nextfloor::objects::Border> MakeBorder(const glm::vec3& location, float scale) const final;
    std::unique_ptr<nextfloor::objects::Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::CollisionEngine> MakeCollisionEngine() const final;

private:
    nextfloor::polygons::PolygonFactory* polygon_factory_;
};

}  // namespace physics

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSICS_MESHPHYSICFACTORY_H_
