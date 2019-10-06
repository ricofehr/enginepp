/**
 *  @file game_loop.h
 *  @brief GameLoop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CONTROLLER_GAMELOOP_H_
#define NEXTFLOOR_CONTROLLER_GAMELOOP_H_

#include "nextfloor/gameplay/loop.h"

#include <list>
#include <memory>

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
class GameLoop : public Loop {

public:
    GameLoop(std::unique_ptr<Level> level,
             SceneWindow* game_window,
             std::unique_ptr<InputHandler> input_handler,
             std::unique_ptr<FrameTimer> timer);
    ~GameLoop() noexcept;

    GameLoop(GameLoop&&) = default;
    GameLoop& operator=(GameLoop&&) = default;
    GameLoop(const GameLoop&) = delete;
    GameLoop& operator=(const GameLoop&) = delete;

    void RunLoop() final;

private:
    void UpdateTime();
    void UpdateCameraOrientation();
    void HandlerInput();
    void Draw();
    void LogLoop();
    void LogFps();
    void PollEvents();
    bool IsNextFrame() const;

    std::unique_ptr<InputHandler> input_handler_{nullptr};
    SceneWindow* game_window_{nullptr};
    std::unique_ptr<FrameTimer> timer_{nullptr};
    std::unique_ptr<Level> level_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GAMELOOP_H_
