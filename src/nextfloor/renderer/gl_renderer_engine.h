/**
 *  @file gl_renderer.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLRENDERER_H_
#define NEXTFLOOR_RENDERER_GLRENDERER_H_

#include "nextfloor/objects/renderer_engine.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <SOIL/SOIL.h>

#include <string>

#include "nextfloor/renderer/scene_window.h"
#include "nextfloor/objects/model_mesh.h"

namespace nextfloor {

namespace renderer {

class GlRendererEngine : public nextfloor::objects::RendererEngine {

public:
    ~GlRendererEngine() override = default;

protected:
    GlRendererEngine(const std::string& texture);

    GlRendererEngine(GlRendererEngine&&) = default;
    GlRendererEngine& operator=(GlRendererEngine&&) = default;
    GlRendererEngine(const GlRendererEngine&) = delete;
    GlRendererEngine& operator=(const GlRendererEngine&) = delete;

    GLuint elementbuffer_;
    GLuint vertexbuffer_;
    GLuint texturebuffer_;
    std::string texture_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLRENDERER_H_