/**
 *  @file game_loop.h
 *  @brief GameLoop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CONTROLLER_GAMELOOP_H_
#define NEXTFLOOR_CONTROLLER_GAMELOOP_H_

#include <list>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/objects/collision_engine.h"
#include "nextfloor/renderer/scene_window.h"
#include "nextfloor/hid/input_handler.h"
#include "nextfloor/core/frame_timer.h"

namespace nextfloor {

namespace controller {

/**
 *  @class GameLoop
 *  @brief  GameLoop manages the lifetime of the opengl scene
 */
class GameLoop {

public:
    GameLoop();
    ~GameLoop() noexcept;

    GameLoop(GameLoop&&) = default;
    GameLoop& operator=(GameLoop&&) = default;
    GameLoop(const GameLoop&) = delete;
    GameLoop& operator=(const GameLoop&) = delete;

    void Loop();

private:
    void SetActiveCamera(nextfloor::objects::Camera* active_camera);
    void UpdateTime();
    void UpdateCameraOrientation();
    void HandlerInput();
    void DrawUniverse();
    void LogLoop();
    void LogFps();

    std::unique_ptr<nextfloor::objects::Mesh> universe_{nullptr};
    std::unique_ptr<nextfloor::hid::InputHandler> input_handler_{nullptr};
    nextfloor::renderer::SceneWindow* game_window_{nullptr};
    nextfloor::objects::Mesh* player_{nullptr};
    std::list<nextfloor::objects::Camera*> game_cameras_;
    std::unique_ptr<nextfloor::core::FrameTimer> timer_{nullptr};
};

}  // namespace controller

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GAMELOOP_H_
