/**
 *  @file gl_cube_renderer.h
 *  @brief Cube Renderer class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLCUBERENDERER_H_
#define NEXTFLOOR_RENDERER_GLCUBERENDERER_H_

#include "nextfloor/renderer/gl_renderer.h"

namespace nextfloor {

namespace renderer {

class GlCubeRenderer : public GlRenderer {

public:

    GlCubeRenderer(std::string texture);

    GlCubeRenderer(GlCubeRenderer&&) = default;
    GlCubeRenderer& operator=(GlCubeRenderer&&) = default;
    GlCubeRenderer(const GlCubeRenderer&) = default;
    GlCubeRenderer& operator=(const GlCubeRenderer&) = default;

    virtual ~GlCubeRenderer() override = default;

    virtual void Draw(const glm::mat4& mvp) noexcept override final;

private:

    void CreateVertexBuffer() noexcept;
    void CreateElementBuffer() noexcept;
    void CreateTextureBuffer() noexcept;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GLCUBERENDERER_H_