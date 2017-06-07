/*
 *  LoopGL class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  LoopGL manage the lifetime of the opengl scene: 
 *      Create the scene
 *      Manages shaders
 *      Loop for display frames
 *      Exit if interrupted by user
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

    /*
     *  Setup the GL Scene
     */
    void InitGL();

    /*
     *  Loop and Record Events
     */
    void Loop(engine::universe::Universe* universe);

    /*
     *  Return sole Instance
     */
    static LoopGL* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new LoopGL;
        return instance;
    }

    /* 
     *  Global variables
     *      GL Window
     *      GL Program
     *      GL Matrix
     */
    static GLFWwindow* sGLWindow;
    static GLuint sProgramId;
    static GLuint sMatrixId;

protected:

    /*
     *  Standard Constructors and Assignments
     *  Protected scope beacause singleton class
     */
    LoopGL(){};
    LoopGL(const LoopGL&) = default;
    LoopGL& operator=(const LoopGL&) = default;
};

} // namespace renderer

} // namespace engine

#endif // ENGINE_RENDERER_LOOPGL_H_
