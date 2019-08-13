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
class MeshFactory {

public:

    virtual ~MeshFactory() = default;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(glm::vec3 location, glm::vec3 scale) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRock(glm::vec3 location) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(glm::vec3 location) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Polygon> MakeCube(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Border> MakeBorder(glm::vec3 location, glm::vec3 scale) const noexcept = 0;

    virtual std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* room_grid) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* universe_grid) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* grid) const noexcept = 0;

    virtual nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::renderer::Shader> MakeVertexShader(std::string shader_path) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::renderer::Shader> MakeFragmentShader(std::string shader_path) const noexcept = 0;
    virtual std::unique_ptr<nextfloor::renderer::SceneWindow> MakeSceneWindow() const noexcept = 0;

    virtual nextfloor::objects::CollisionEngine* MakeCollisionEngine() const noexcept = 0;

    virtual std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const noexcept = 0;

protected:

    MeshFactory() = default;

    MeshFactory(MeshFactory&&) = default;
    MeshFactory& operator=(MeshFactory&&) = default;
    MeshFactory(const MeshFactory&) = default;
    MeshFactory& operator=(const MeshFactory&) = default;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_FACTORY_H_
