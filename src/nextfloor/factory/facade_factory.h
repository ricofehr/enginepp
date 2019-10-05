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
#include "nextfloor/gameplay/level.h"

#include "nextfloor/objects/renderer_engine.h"
#include "nextfloor/renderer/shader.h"
#include "nextfloor/gameplay/scene_window.h"
#include "nextfloor/renderer/scene_input.h"

#include "nextfloor/gameplay/hid.h"
#include "nextfloor/gameplay/input_handler.h"

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"

#include "nextfloor/gameplay/action.h"

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
    ~FacadeFactory() noexcept;

    FacadeFactory(FacadeFactory&&) = default;
    FacadeFactory& operator=(FacadeFactory&&) = default;
    FacadeFactory(const FacadeFactory&) = delete;
    FacadeFactory& operator=(const FacadeFactory&) = delete;

    /* MeshFactory Part */
    std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() const;

    std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(const glm::vec3& location) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(const glm::vec3& location,
                                                            const glm::vec3& scale,
                                                            const std::string& texture) const;

    std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(const glm::vec3& location) const;

    std::unique_ptr<nextfloor::objects::Mesh> MakeRock(const glm::vec3& location) const;
    std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(const glm::vec3& location) const;

    std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) const;

    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const;

    std::unique_ptr<nextfloor::objects::Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const;

    std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) const;
    std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) const;
    std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner,
                                                       const glm::ivec3& boxes_count,
                                                       const glm::vec3& box_dimension) const;
    std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(const glm::vec3& grid_coords,
                                                                 nextfloor::objects::Grid* room_grid) const;
    std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(const glm::vec3& grid_coords,
                                                                     nextfloor::objects::Grid* universe_grid) const;
    std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(const glm::vec3& grid_coords,
                                                             nextfloor::objects::Grid* grid) const;

    /* RendererFactory Part */
    nextfloor::objects::RendererEngine* MakeCubeRenderer(const std::string& texture);
    nextfloor::renderer::Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id);
    nextfloor::renderer::Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id);
    nextfloor::gameplay::SceneWindow* MakeSceneWindow();
    std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput();

    /* PhysicsFactory Part */
    nextfloor::objects::CollisionEngine* MakeCollisionEngine() const;

    /* GameplayFactory Part */
    std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const;

    /* HidFactory Part */
    std::unique_ptr<nextfloor::gameplay::HID> MakeHid() const;
    std::unique_ptr<nextfloor::gameplay::InputHandler> MakeInputHandler() const;

    /* CoreFacotry Part */
    std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() const;
    std::unique_ptr<nextfloor::core::Exit> MakeExit() const;
    std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() const;
    std::unique_ptr<nextfloor::gameplay::FrameTimer> MakeFrameTimer() const;
    std::unique_ptr<nextfloor::core::Log> MakeLog() const;
    std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() const;
    std::unique_ptr<MeshFactory> MakeMeshFactory() const;
    std::unique_ptr<HidFactory> MakeHidFactory() const;
    std::unique_ptr<nextfloor::core::WindowSettings> MakeWindowSettings(nextfloor::gameplay::SceneWindow* window) const;

    /* CommandFactory Part */
    std::unique_ptr<nextfloor::gameplay::Action> MakeFireAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeJumpAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveDownAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveLeftAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveRightAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveUpAction() const;
    std::unique_ptr<nextfloor::gameplay::Action> MakeRunAction() const;

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
