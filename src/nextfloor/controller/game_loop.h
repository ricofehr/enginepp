/**
 *  @file game_loop.h
 *  @brief GameLoop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CONTROLLER_GAMELOOP_H_
#define NEXTFLOOR_CONTROLLER_GAMELOOP_H_

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/collision_engine.h"
#include "nextfloor/renderer/scene_window.h"

namespace nextfloor {

namespace controller {

/**
 *  @class GameLoop
 *  @brief  GameLoop manages the lifetime of the opengl scene
 */
class GameLoop {

public:
    GameLoop();

    GameLoop(GameLoop&&) = default;
    GameLoop& operator=(GameLoop&&) = default;
    GameLoop(const GameLoop&) = delete;
    GameLoop& operator=(const GameLoop&) = delete;

    ~GameLoop();

    void Loop();

private:
    void LogLoop();
    void LogFps();

    std::unique_ptr<nextfloor::objects::Mesh> universe_{nullptr};
    nextfloor::renderer::SceneWindow* game_window_{nullptr};
};

}  // namespace controller

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GAMELOOP_H_
