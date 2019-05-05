/**
 *  @file game_loop.h
 *  @brief GameLoop class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_JOB_GAMELOOP_H_
#define NEXTFLOOR_JOB_GAMELOOP_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "nextfloor/universe/universe.h"
#include "nextfloor/renderer/game_window.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::renderer
 *  @brief Prepare, Render, and Display GL Scene
 */
namespace job {

/**
 *  @class GameLoop
 *  @brief  GameLoop manages the lifetime of the opengl scene
 */
class GameLoop {

public:

    /**
     *  Constructor, ensure only one instance is created
     */
    GameLoop(nextfloor::renderer::GameWindow* game_window);

    /**
     *  Default Move constructor
     */
    GameLoop(GameLoop&&) = default;

    /**
     *  Default Move assignment
     */
    GameLoop& operator=(GameLoop&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    GameLoop(const GameLoop&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    GameLoop& operator=(const GameLoop&) = delete;

    /**
     *  Destructor - reset instanciated flag
     */
    ~GameLoop();

    /**
     *  Loop and Record Events
     *  @param universe is The universe of the program
     */
    void Loop(nextfloor::universe::Universe* universe);

private:

    void LoopLog(nextfloor::universe::Universe* universe);

    nextfloor::renderer::GameWindow* game_window_{nullptr};

};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GAMELOOP_H_
