/**
 *  @file game_loop.h
 *  @brief GameLoop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CONTROLLER_GAMELOOP_H_
#define NEXTFLOOR_CONTROLLER_GAMELOOP_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "nextfloor/objects/universe.h"
#include "nextfloor/objects/collision_engine.h"
#include "nextfloor/renderer/game_window.h"

namespace nextfloor {

namespace controller {

/**
 *  @class GameLoop
 *  @brief  GameLoop manages the lifetime of the opengl scene
 */
class GameLoop {

public:

    GameLoop(nextfloor::renderer::GameWindow* game_window,
             nextfloor::objects::CollisionEngine* engine_collision);

    GameLoop(GameLoop&&) = default;

    GameLoop& operator=(GameLoop&&) = default;

    /* Copy constructor Deleted : Ensure a sole Instance */
    GameLoop(const GameLoop&) = delete;

    /* Copy assignment Deleted : Ensure a sole Instance */
    GameLoop& operator=(const GameLoop&) = delete;

    ~GameLoop();

    void Loop(nextfloor::objects::Mesh* universe);

private:

    void LoopLog(nextfloor::objects::Mesh* universe);

    nextfloor::renderer::GameWindow* game_window_{nullptr};

    nextfloor::objects::CollisionEngine* engine_collision_{nullptr};
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GAMELOOP_H_
