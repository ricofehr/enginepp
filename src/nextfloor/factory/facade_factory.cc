/**
 *  @file facade_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/facade_factory.h"

#include "nextfloor/factory/game_factory.h"
#include "nextfloor/factory/gl_renderer_factory.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

FacadeFactory::FacadeFactory()
{
    assert(!sInstanciated);
    mesh_factory_ = std::make_unique<GameFactory>();
    renderer_factory_ = std::make_unique<GlRendererFactory>();
    sInstanciated = true;
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeUniverse() const
{
    return mesh_factory_->MakeUniverse();
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRoom(const glm::vec3& location) const
{
    return mesh_factory_->MakeRoom(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeFrontWall(const glm::vec3& location,
                                                                       const glm::vec3& scale) const
{
    return mesh_factory_->MakeFrontWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRightWall(const glm::vec3& location,
                                                                       const glm::vec3& scale) const
{
    return mesh_factory_->MakeRightWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeBackWall(const glm::vec3& location,
                                                                      const glm::vec3& scale) const
{
    return mesh_factory_->MakeBackWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeLeftWall(const glm::vec3& location,
                                                                      const glm::vec3& scale) const
{
    return mesh_factory_->MakeLeftWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeFloor(const glm::vec3& location, const glm::vec3& scale) const
{
    return mesh_factory_->MakeFloor(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRoof(const glm::vec3& location, const glm::vec3& scale) const
{
    return mesh_factory_->MakeRoof(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeWallBrick(const glm::vec3& location,
                                                                       const glm::vec3& scale,
                                                                       const std::string& texture) const
{
    return mesh_factory_->MakeWallBrick(location, scale, texture);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakePlayer(const glm::vec3& location) const
{
    return mesh_factory_->MakePlayer(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRock(const glm::vec3& location) const
{
    return mesh_factory_->MakeRock(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeLittleRock(const glm::vec3& location) const
{
    return mesh_factory_->MakeLittleRock(location);
}

std::unique_ptr<nextfloor::objects::Camera> FacadeFactory::MakeCamera(nextfloor::objects::Mesh* owner) const
{
    return mesh_factory_->MakeCamera(owner);
}

std::unique_ptr<nextfloor::objects::Polygon> FacadeFactory::MakeCube(const glm::vec3& location, const glm::vec3& scale) const
{
    return mesh_factory_->MakeCube(location, scale);
}


std::unique_ptr<nextfloor::objects::Polygon> FacadeFactory::MakeCube(const glm::vec3& location,
                                                                     const glm::vec3& scale,
                                                                     const std::string& texture) const
{
    return mesh_factory_->MakeCube(location, scale, texture);
}

std::unique_ptr<nextfloor::objects::Border> FacadeFactory::MakeBorder(const glm::vec3& location,
                                                                      const glm::vec3& scale) const
{
    return mesh_factory_->MakeBorder(location, scale);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeUniverseGrid(nextfloor::objects::Mesh* universe) const
{
    return mesh_factory_->MakeUniverseGrid(universe);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeRoomGrid(nextfloor::objects::Mesh* room) const
{
    return mesh_factory_->MakeRoomGrid(room);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeGrid(nextfloor::objects::Mesh* owner,
                                                                  const glm::ivec3& boxes_count,
                                                                  const glm::vec3& box_dimension) const
{
    return mesh_factory_->MakeGrid(owner, boxes_count, box_dimension);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeRoomGridBox(const glm::vec3& coords,
                                                                            nextfloor::objects::Grid* room_grid) const
{
    return mesh_factory_->MakeRoomGridBox(coords, room_grid);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeUniverseGridBox(const glm::vec3& coords,
                                                                                nextfloor::objects::Grid* universe_grid) const
{
    return mesh_factory_->MakeUniverseGridBox(coords, universe_grid);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeGridBox(const glm::vec3& coords,
                                                                        nextfloor::objects::Grid* grid) const
{
    return mesh_factory_->MakeUniverseGridBox(coords, grid);
}

nextfloor::gameplay::RendererEngine* FacadeFactory::MakeCubeRenderer(const std::string& texture)
{
    return renderer_factory_->MakeCubeRenderer(texture);
}

nextfloor::renderer::Shader* FacadeFactory::MakeVertexShader(const std::string& shader_path, unsigned int program_id)
{
    return renderer_factory_->MakeVertexShader(shader_path, program_id);
}

nextfloor::renderer::Shader* FacadeFactory::MakeFragmentShader(const std::string& shader_path, unsigned int program_id)
{
    return renderer_factory_->MakeFragmentShader(shader_path, program_id);
}

nextfloor::gameplay::SceneWindow* FacadeFactory::MakeSceneWindow()
{
    return renderer_factory_->MakeSceneWindow();
}

std::unique_ptr<nextfloor::renderer::SceneInput> FacadeFactory::MakeSceneInput()
{
    return renderer_factory_->MakeSceneInput();
}

nextfloor::objects::CollisionEngine* FacadeFactory::MakeCollisionEngine() const
{
    return mesh_factory_->MakeCollisionEngine();
}

std::unique_ptr<nextfloor::gameplay::Level> FacadeFactory::MakeLevel() const
{
    return mesh_factory_->MakeLevel();
}

FacadeFactory::~FacadeFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor