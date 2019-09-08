/**
 *  @file facade_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_FACADEFACTORY_H_
#define NEXTFLOOR_FACTORY_FACADEFACTORY_H_

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

#include "nextfloor/objects/renderer_engine.h"
#include "nextfloor/renderer/shader.h"
#include "nextfloor/renderer/scene_window.h"
#include "nextfloor/renderer/scene_input.h"

#include "nextfloor/hid/hid.h"
#include "nextfloor/hid/input_handler.h"

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/timer.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"

#include "nextfloor/actions/action.h"

#include "nextfloor/factory/renderer_factory.h"
#include "nextfloor/factory/mesh_factory.h"
#include "nextfloor/factory/hid_factory.h"
#include "nextfloor/factory/core_factory.h"
#include "nextfloor/factory/action_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class FacadeFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class FacadeFactory {

public:
    FacadeFactory();

    FacadeFactory(FacadeFactory&&) = default;
    FacadeFactory& operator=(FacadeFactory&&) = default;
    FacadeFactory(const FacadeFactory&) = default;
    FacadeFactory& operator=(const FacadeFactory&) = default;

    ~FacadeFactory();

    /* MeshFactory Part */
    std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse();

    std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(glm::vec3 location);
    std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(glm::vec3 location, glm::vec3 scale);
    std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture);

    std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(glm::vec3 location);

    std::unique_ptr<nextfloor::objects::Mesh> MakeRock(glm::vec3 location);
    std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(glm::vec3 location);

    std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner);

    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(glm::vec3 location, glm::vec3 scale);

    std::unique_ptr<nextfloor::objects::Border> MakeBorder(glm::vec3 location, glm::vec3 scale);

    std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe);
    std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room);
    std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner,
                                                       glm::ivec3 boxes_count,
                                                       glm::vec3 box_dimension);
    std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(glm::vec3 grid_coords,
                                                                 nextfloor::objects::Grid* room_grid);
    std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(glm::vec3 grid_coords,
                                                                     nextfloor::objects::Grid* universe_grid);
    std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* grid);

    /* RendererFactory Part */
    nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture);
    nextfloor::renderer::Shader* MakeVertexShader(std::string shader_path, unsigned int program_id);
    nextfloor::renderer::Shader* MakeFragmentShader(std::string shader_path, unsigned int program_id);
    nextfloor::renderer::SceneWindow* MakeSceneWindow();
    std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput();

    /* PhysicsFactory Part */
    nextfloor::objects::CollisionEngine* MakeCollisionEngine();

    /* GameplayFactory Part */
    std::unique_ptr<nextfloor::gameplay::Level> MakeLevel();

    /* HidFactory Part */
    std::unique_ptr<nextfloor::hid::HID> MakeHid();
    std::unique_ptr<nextfloor::hid::InputHandler> MakeInputHandler();

    /* CoreFacotry Part */
    std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser();
    std::unique_ptr<nextfloor::core::Exit> MakeExit();
    std::unique_ptr<nextfloor::core::FileIO> MakeFileIO();
    std::unique_ptr<nextfloor::core::Timer> MakeTimer();
    std::unique_ptr<nextfloor::core::Log> MakeLog();
    std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator();
    std::unique_ptr<MeshFactory> MakeMeshFactory();
    std::unique_ptr<HidFactory> MakeHidFactory();
    std::unique_ptr<nextfloor::core::WindowSettings> MakeWindowSettings(nextfloor::renderer::SceneWindow* window);

    /* CommandFactory Part */
    std::unique_ptr<nextfloor::actions::Action> MakeFireAction();
    std::unique_ptr<nextfloor::actions::Action> MakeJumpAction();
    std::unique_ptr<nextfloor::actions::Action> MakeMoveDownAction();
    std::unique_ptr<nextfloor::actions::Action> MakeMoveLeftAction();
    std::unique_ptr<nextfloor::actions::Action> MakeMoveRightAction();
    std::unique_ptr<nextfloor::actions::Action> MakeMoveUpAction();
    std::unique_ptr<nextfloor::actions::Action> MakeRunAction();

private:
    std::unique_ptr<MeshFactory> mesh_factory_;
    std::unique_ptr<RendererFactory> renderer_factory_;
    std::unique_ptr<CoreFactory> core_factory_;
    std::unique_ptr<HidFactory> hid_factory_;
    std::unique_ptr<CommandFactory> command_factory_;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_FACADEFACTORY_H_
