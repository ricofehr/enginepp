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
#include "nextfloor/renderer/shader.h"
#include "nextfloor/renderer/scene_window.h"
#include "nextfloor/gameplay/level.h"

namespace nextfloor {

namespace factory {

/**
 *  @class MeshFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class MeshFactory {  // clang-format off

public:
    virtual ~MeshFactory() = default;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(glm::vec3 location) = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(glm::vec3 location, glm::vec3 scale) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(glm::vec3 location,
                                                                    glm::vec3 scale,
                                                                    std::string texture) = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(glm::vec3 location) = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRock(glm::vec3 location) = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(glm::vec3 location) = 0;

    virtual std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) = 0;

    virtual std::unique_ptr<nextfloor::objects::Polygon> MakeCube(glm::vec3 location, glm::vec3 scale) = 0;

    virtual std::unique_ptr<nextfloor::objects::Border> MakeBorder(glm::vec3 location, glm::vec3 scale) = 0;

    virtual std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner,
                                                               glm::ivec3 boxes_count,
                                                               glm::vec3 box_dimension) = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(glm::vec3 coords,
                                                                         nextfloor::objects::Grid* grid) = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(glm::vec3 coords,
                                                                             nextfloor::objects::Grid* grid) = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(glm::vec3 coords,
                                                                     nextfloor::objects::Grid* grid) = 0;

    virtual nextfloor::objects::CollisionEngine* MakeCollisionEngine() = 0;

    virtual std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() = 0;

protected:
    MeshFactory() = default;

    MeshFactory(MeshFactory&&) = default;
    MeshFactory& operator=(MeshFactory&&) = default;
    MeshFactory(const MeshFactory&) = default;
    MeshFactory& operator=(const MeshFactory&) = default;
};  // clang-format on

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_MESHFACTORY_H_
