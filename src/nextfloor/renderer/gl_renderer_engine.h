/**
 *  @file gl_renderer.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLRENDERER_H_
#define NEXTFLOOR_RENDERER_GLRENDERER_H_

#include "nextfloor/gameplay/renderer_engine.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

namespace nextfloor {

namespace renderer {

class GlRendererEngine : public nextfloor::gameplay::RendererEngine {

public:
    ~GlRendererEngine() override = default;

protected:
    GlRendererEngine(const std::string& texture, GLuint program_id, GLuint matrix_id);

    GlRendererEngine(GlRendererEngine&&) = default;
    GlRendererEngine& operator=(GlRendererEngine&&) = default;
    GlRendererEngine(const GlRendererEngine&) = delete;
    GlRendererEngine& operator=(const GlRendererEngine&) = delete;

    GLuint program_id_;
    GLuint matrix_id_;

    GLuint elementbuffer_;
    GLuint vertexbuffer_;
    GLuint texturebuffer_;
    std::string texture_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLRENDERER_H_