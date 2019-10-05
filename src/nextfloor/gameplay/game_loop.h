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
#include "nextfloor/gameplay/scene_window.h"
#include "nextfloor/gameplay/input_handler.h"
#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/gameplay/level.h"

namespace nextfloor {

namespace gameplay {

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
    std::unique_ptr<InputHandler> input_handler_{nullptr};
    SceneWindow* game_window_{nullptr};
    nextfloor::objects::Mesh* player_{nullptr};
    std::list<nextfloor::objects::Camera*> game_cameras_;
    std::unique_ptr<FrameTimer> timer_{nullptr};
    std::unique_ptr<Level> level_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GAMELOOP_H_
