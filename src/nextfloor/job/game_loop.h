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
 *  @brief  GameLoop manages the lifetime of the opengl scene\n
 *  Implement Singleton Design Pattern which ensure a sole GameLoop object for the program
 */
class GameLoop {

public:

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
     *  Return (and allocates if needed) sole Instance
     *  @return the sole GameLoop instance
     */
    static GameLoop* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new GameLoop;
        return instance;
    }

    /**
     *  Setup the GL Scene
     */
    void InitGL();

    /**
     *  Loop and Record Events
     *  @param universe is The universe of the program
     */
    void Loop(nextfloor::universe::Universe* universe);

    /** A Global variable for the GL Matrix */
    static GLuint sMatrixId;

private:

    /**
     *  Default Constructor
     *  Protected scope beacause singleton class
     */
    GameLoop() = default;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GAMELOOP_H_
