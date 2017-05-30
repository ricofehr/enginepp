/*
 * LoopGL class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Implement Singleton Design Pattern which ensure a sole LoopGL object for the program
 */

#ifndef ENGINE_RENDERER_LOOPGL_H_
#define ENGINE_RENDERER_LOOPGL_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace engine {

/* Forward declaration of class Universe */
namespace universe {
    class Universe;
}//namespace universe

namespace renderer {

class LoopGL {

public:

    void InitGL();
    void Loop(engine::universe::Universe *universe);

    static LoopGL *Instance()
    {
        /* Raw pointer because static var */
        static auto *instance = new LoopGL;
        return instance;
    }

    /* Global Static variables */
    static GLFWwindow *sGLWindow;
    static GLuint sProgramId;
    static GLuint sMatrixId;

protected:

    LoopGL(){};
    LoopGL(const LoopGL&) = default;
    LoopGL& operator=(const LoopGL&) = default;
};

}//namespace renderer
}//namespace engine

#endif //ENGINE_RENDERER_LOOPGL_H_
