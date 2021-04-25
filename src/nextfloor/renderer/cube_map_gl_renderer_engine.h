/**
 *  @file cube_map_gl_renderer.h
 *  @brief CubeMap GL Renderer class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_CUBEMAPGLRENDERER_H_
#define NEXTFLOOR_RENDERER_CUBEMAPGLRENDERER_H_

#include "nextfloor/renderer/gl_renderer_engine.h"

#include <string>
#include <GL/glew.h>

namespace nextfloor {

namespace renderer {

class CubeMapGlRendererEngine : public GlRendererEngine {

public:
    CubeMapGlRendererEngine(PipelineProgram* pipeline_program);
    ~CubeMapGlRendererEngine() final = default;

    void Draw(const glm::mat4& mvp) final;

private:
    void Init();
    void CreateVertexBuffer();
    void CreateTextureBuffer();

    bool is_initialized_ = false;

    GLuint vertexbuffer_;
    GLuint vertexarray_;
    GLuint texturebuffer_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_CUBEMAPGLRENDERER_H_
