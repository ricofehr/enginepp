/**
 *  @file mesh_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_MESHFACTORY_H_
#define NEXTFLOOR_FACTORY_MESHFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/border.h"
#include "nextfloor/objects/renderer_engine.h"
#include "nextfloor/objects/collision_engine.h"
#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/grid_box.h"
#include "nextfloor/objects/polygon.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/gameplay/level.h"

namespace nextfloor {

namespace factory {

/**
 *  @class MeshFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class MeshFactory {

public:
    virtual ~MeshFactory() = default;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(const glm::vec3& location) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(const glm::vec3& location,
                                                                    const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(const glm::vec3& location,
                                                                    const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(const glm::vec3& location,
                                                                   const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(const glm::vec3& location,
                                                                   const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(const glm::vec3& location,
                                                                const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(const glm::vec3& location,
                                                               const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(const glm::vec3& location,
                                                                    const glm::vec3& scale,
                                                                    const std::string& texture) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(const glm::vec3& location) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRock(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(const glm::vec3& location) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location,
                                                                  const glm::vec3& scale) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Border> MakeBorder(const glm::vec3& location,
                                                                   const glm::vec3& scale) const = 0;

    virtual std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) const = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner,
                                                               const glm::ivec3& boxes_count,
                                                               const glm::vec3& box_dimension) const = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(const glm::vec3& coords,
                                                                         nextfloor::objects::Grid* grid) const = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(const glm::vec3& coords,
                                                                             nextfloor::objects::Grid* grid) const = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(const glm::vec3& coords,
                                                                     nextfloor::objects::Grid* grid) const = 0;

    virtual nextfloor::objects::CollisionEngine* MakeCollisionEngine() const = 0;

    virtual std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const = 0;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_MESHFACTORY_H_
