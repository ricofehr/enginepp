/**
 *  @file facade_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/facade_factory.h"

#include "nextfloor/factory/game_factory.h"
#include "nextfloor/factory/gl_renderer_factory.h"
#include "nextfloor/factory/services_factory.h"
#include "nextfloor/factory/mouse_hid_factory.h"
#include "nextfloor/factory/action_factory.h"

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
    core_factory_ = std::make_unique<ServicesFactory>();
    hid_factory_ = std::make_unique<MouseHidFactory>();
    command_factory_ = std::make_unique<ActionFactory>();
    sInstanciated = true;
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeUniverse()
{
    return mesh_factory_->MakeUniverse();
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRoom(glm::vec3 location)
{
    return mesh_factory_->MakeRoom(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeFrontWall(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeFrontWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRightWall(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeRightWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeBackWall(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeBackWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeLeftWall(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeLeftWall(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeFloor(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeFloor(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRoof(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeRoof(location, scale);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeWallBrick(glm::vec3 location,
                                                                       glm::vec3 scale,
                                                                       std::string texture)
{
    return mesh_factory_->MakeWallBrick(location, scale, texture);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakePlayer(glm::vec3 location)
{
    return mesh_factory_->MakePlayer(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeRock(glm::vec3 location)
{
    return mesh_factory_->MakeRock(location);
}

std::unique_ptr<nextfloor::objects::Mesh> FacadeFactory::MakeLittleRock(glm::vec3 location)
{
    return mesh_factory_->MakeLittleRock(location);
}

std::unique_ptr<nextfloor::objects::Camera> FacadeFactory::MakeCamera(nextfloor::objects::Mesh* owner)
{
    return mesh_factory_->MakeCamera(owner);
}

std::unique_ptr<nextfloor::objects::Polygon> FacadeFactory::MakeCube(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeCube(location, scale);
}

std::unique_ptr<nextfloor::objects::Border> FacadeFactory::MakeBorder(glm::vec3 location, glm::vec3 scale)
{
    return mesh_factory_->MakeBorder(location, scale);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeUniverseGrid(nextfloor::objects::Mesh* universe)
{
    return mesh_factory_->MakeUniverseGrid(universe);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeRoomGrid(nextfloor::objects::Mesh* room)
{
    return mesh_factory_->MakeRoomGrid(room);
}

std::unique_ptr<nextfloor::objects::Grid> FacadeFactory::MakeGrid(nextfloor::objects::Mesh* owner,
                                                                  glm::ivec3 boxes_count,
                                                                  glm::vec3 box_dimension)
{
    return mesh_factory_->MakeGrid(owner, boxes_count, box_dimension);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeRoomGridBox(glm::vec3 coords,
                                                                            nextfloor::objects::Grid* room_grid)
{
    return mesh_factory_->MakeRoomGridBox(coords, room_grid);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeUniverseGridBox(glm::vec3 coords,
                                                                                nextfloor::objects::Grid* universe_grid)
{
    return mesh_factory_->MakeUniverseGridBox(coords, universe_grid);
}

std::unique_ptr<nextfloor::objects::GridBox> FacadeFactory::MakeGridBox(glm::vec3 coords, nextfloor::objects::Grid* grid)
{
    return mesh_factory_->MakeUniverseGridBox(coords, grid);
}

nextfloor::objects::RendererEngine* FacadeFactory::MakeCubeRenderer(std::string texture)
{
    return renderer_factory_->MakeCubeRenderer(texture);
}

nextfloor::renderer::Shader* FacadeFactory::MakeVertexShader(std::string shader_path)
{
    return renderer_factory_->MakeVertexShader(shader_path);
}

nextfloor::renderer::Shader* FacadeFactory::MakeFragmentShader(std::string shader_path)
{
    return renderer_factory_->MakeFragmentShader(shader_path);
}

nextfloor::renderer::SceneWindow* FacadeFactory::MakeSceneWindow()
{
    return renderer_factory_->MakeSceneWindow();
}

nextfloor::objects::CollisionEngine* FacadeFactory::MakeCollisionEngine()
{
    return mesh_factory_->MakeCollisionEngine();
}

std::unique_ptr<nextfloor::gameplay::Level> FacadeFactory::MakeLevel()
{
    return mesh_factory_->MakeLevel();
}

std::unique_ptr<nextfloor::hid::InputHandler> FacadeFactory::MakeInputHandler()
{
    return hid_factory_->MakeInputHandler();
}

std::unique_ptr<nextfloor::hid::HID> FacadeFactory::MakeHid()
{
    return hid_factory_->MakeHid();
}

std::unique_ptr<nextfloor::core::ConfigParser> FacadeFactory::MakeConfigParser()
{
    return core_factory_->MakeConfigParser();
}

std::unique_ptr<nextfloor::core::Exit> FacadeFactory::MakeExit()
{
    return core_factory_->MakeExit();
}

std::unique_ptr<nextfloor::core::FileIO> FacadeFactory::MakeFileIO()
{
    return core_factory_->MakeFileIO();
}

std::unique_ptr<nextfloor::core::Timer> FacadeFactory::MakeTimer()
{
    return core_factory_->MakeTimer();
}

std::unique_ptr<nextfloor::core::Log> FacadeFactory::MakeLog()
{
    return core_factory_->MakeLog();
}

std::unique_ptr<nextfloor::core::RandomGenerator> FacadeFactory::MakeRandomGenerator()
{
    return core_factory_->MakeRandomGenerator();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeFireAction()
{
    return command_factory_->MakeFireAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeJumpAction()
{
    return command_factory_->MakeJumpAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeMoveDownAction()
{
    return command_factory_->MakeMoveDownAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeMoveLeftAction()
{
    return command_factory_->MakeMoveLeftAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeMoveRightAction()
{
    return command_factory_->MakeMoveRightAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeMoveUpAction()
{
    return command_factory_->MakeMoveUpAction();
}

std::unique_ptr<nextfloor::actions::Action> FacadeFactory::MakeRunAction()
{
    return command_factory_->MakeRunAction();
}

FacadeFactory::~FacadeFactory()
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor