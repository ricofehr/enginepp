/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_MODELMESHFACTORY_H_
#define NEXTFLOOR_PLAYGROUND_MODELMESHFACTORY_H_

#include "nextfloor/playground/ground_factory.h"

#include <glm/glm.hpp>

#include "nextfloor/playground/wall.h"

#include "nextfloor/playground/grid_factory.h"
#include "nextfloor/mesh/border_factory.h"
#include "nextfloor/thing/thing_factory.h"

namespace nextfloor {

namespace playground {

/**
 *  @class GameGroundFactory
 *  @brief Concrete factory for Universe Objects
 */
class GameGroundFactory : public GroundFactory {

public:
    GameGroundFactory(nextfloor::thing::ThingFactory* thing_factory,
                      GridFactory* grid_factory,
                      nextfloor::mesh::BorderFactory* border_factory);
    ~GameGroundFactory() final = default;

    std::unique_ptr<Ground> MakeUniverse(std::vector<std::unique_ptr<Ground>> rooms) const final;
    std::unique_ptr<Ground> MakeRoom(const glm::vec3& location,
                                     std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects) const final;

private:
    std::unique_ptr<Wall> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<Wall> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<Wall> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<Wall> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<Wall> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<Wall> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const;

    nextfloor::thing::ThingFactory* thing_factory_;
    GridFactory* grid_factory_;
    nextfloor::mesh::BorderFactory* border_factory_;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_GAMEGROUNDFACTORY_H_
