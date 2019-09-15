/**
 *  @file cube_gl_renderer.h
 *  @brief Cube GL Renderer class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_CUBEGLRENDERER_H_
#define NEXTFLOOR_RENDERER_CUBEGLRENDERER_H_

#include "nextfloor/renderer/gl_renderer_engine.h"

namespace nextfloor {

namespace renderer {

class CubeGlRendererEngine : public GlRendererEngine {

public:
    CubeGlRendererEngine(std::string texture);

    CubeGlRendererEngine(CubeGlRendererEngine&&) = default;
    CubeGlRendererEngine& operator=(CubeGlRendererEngine&&) = default;
    CubeGlRendererEngine(const CubeGlRendererEngine&) = default;
    CubeGlRendererEngine& operator=(const CubeGlRendererEngine&) = default;

    virtual ~CubeGlRendererEngine() override = default;

    virtual void Draw(const glm::mat4& mvp) noexcept override final;

private:
    void Init();
    void CreateVertexBuffer() noexcept;
    void CreateElementBuffer() noexcept;
    void CreateTextureBuffer() noexcept;

    bool is_initialized_ = false;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_CUBEGLRENDERER_H_