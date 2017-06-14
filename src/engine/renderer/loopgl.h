/**
 *  LoopGL class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *  LoopGL manage the lifetime of the opengl scene\n
 *      Create the scene\n
 *      Manages shaders\n
 *      Loop for display frames\n
 *      Exit if interrupted by user\n\n
 *
 *  Implement Singleton Design Pattern which ensure a sole LoopGL object for the program
 */

#ifndef ENGINE_RENDERER_LOOPGL_H_
#define ENGINE_RENDERER_LOOPGL_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "engine/universe/universe.h"

namespace engine {

namespace renderer {

class LoopGL {

public:

    /**
     *  Default Move constructor
     */
    LoopGL(LoopGL&&) = default;

    /**
     *  Default Move assignment
     */
    LoopGL& operator=(LoopGL&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    LoopGL(const LoopGL&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    LoopGL& operator=(const LoopGL&) = delete;

    /**
     *  Setup the GL Scene
     */
    void InitGL();

    /**
     *  Loop and Record Events
     *  @param universe is The universe of the program
     */
    void Loop(engine::universe::Universe* universe);

    /**
     *  Return (and allocates if needed) sole Instance
     *  @return the sole LoopGL instance
     */
    static LoopGL* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new LoopGL;
        return instance;
    }


    /** A Global variable, GL Window */
    static GLFWwindow* sGLWindow;

    /** A Global variable for the GL Program */
    static GLuint sProgramId;

    /** A Global variable for the GL Matrix */
    static GLuint sMatrixId;

protected:

    /**
     *  Default Constructor
     *  Protected scope beacause singleton class
     */
    LoopGL() = default;
};

} // namespace renderer

} // namespace engine

#endif // ENGINE_RENDERER_LOOPGL_H_
