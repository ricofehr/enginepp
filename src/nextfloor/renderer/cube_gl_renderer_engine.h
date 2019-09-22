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
    CubeGlRendererEngine(const std::string& texture);
    ~CubeGlRendererEngine() final = default;

    void Draw(const glm::mat4& mvp) final;

private:
    void Init();
    void CreateVertexBuffer();
    void CreateElementBuffer();
    void CreateTextureBuffer();

    bool is_initialized_ = false;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_CUBEGLRENDERER_H_