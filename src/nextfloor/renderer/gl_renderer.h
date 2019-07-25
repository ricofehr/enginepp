/**
 *  @file gl_renderer.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLRENDERER_H_
#define NEXTFLOOR_RENDERER_GLRENDERER_H_

#include "nextfloor/objects/engine_renderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <SOIL/SOIL.h>

#include <string>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace renderer {

class GlRenderer : public nextfloor::objects::EngineRenderer {

public:

    GlRenderer(std::string texture)
    {
        texture_ = texture;
    }

    GlRenderer(GlRenderer&&) = default;
    GlRenderer& operator=(GlRenderer&&) = default;
    GlRenderer(const GlRenderer&) = default;
    GlRenderer& operator=(const GlRenderer&) = default;

    virtual ~GlRenderer() override = default;

protected:

    GLuint elementbuffer_;
    GLuint vertexbuffer_;
    GLuint texturebuffer_;
    std::string texture_;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GLRENDERER_H_