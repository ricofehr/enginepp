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
 *  @brief  GameLoop manages the lifetime of the opengl scene
 */
class GameLoop {

public:

    /**
     *  Constructor, ensure only one instance is created
     */
    GameLoop()
    {
        assert(!sInstanciated);
        sInstanciated = true;
    }

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
    ~GameLoop()
    {
        assert(sInstanciated);
        sInstanciated = false;
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
     *  Flag to ensure only one object is created
     */
    static bool sInstanciated;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GAMELOOP_H_
