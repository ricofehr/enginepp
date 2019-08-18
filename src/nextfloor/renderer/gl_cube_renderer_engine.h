/**
 *  @file gl_cube_renderer.h
 *  @brief Cube Renderer class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLCUBERENDERER_H_
#define NEXTFLOOR_RENDERER_GLCUBERENDERER_H_

#include "nextfloor/renderer/gl_renderer_engine.h"

#include <tbb/mutex.h>

namespace nextfloor {

namespace renderer {

class GlCubeRendererEngine : public GlRendererEngine {

public:

    GlCubeRendererEngine(std::string texture);

    GlCubeRendererEngine(GlCubeRendererEngine&&) = default;
    GlCubeRendererEngine& operator=(GlCubeRendererEngine&&) = default;
    GlCubeRendererEngine(const GlCubeRendererEngine&) = default;
    GlCubeRendererEngine& operator=(const GlCubeRendererEngine&) = default;

    virtual ~GlCubeRendererEngine() override = default;

    virtual void Draw(const glm::mat4& mvp) noexcept override final;

private:

    tbb::mutex mutex_;

    void CreateVertexBuffer() noexcept;
    void CreateElementBuffer() noexcept;
    void CreateTextureBuffer() noexcept;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_GLCUBERENDERER_H_