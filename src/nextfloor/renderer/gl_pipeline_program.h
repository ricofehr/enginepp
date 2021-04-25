/**
 *  @file gl_scene_window.h
 *  @brief GL Scene Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLPIPELINEPROGRAM_H_
#define NEXTFLOOR_RENDERER_GLPIPELINEPROGRAM_H_

#include "nextfloor/renderer/pipeline_program.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

#include "nextfloor/renderer/shader.h"
#include "nextfloor/renderer/shader_factory.h"

namespace nextfloor {

namespace renderer {

class GlPipelineProgram : public nextfloor::renderer::PipelineProgram {

public:
    GlPipelineProgram(std::string label, ShaderFactory* shader_factory);
    ~GlPipelineProgram() noexcept override = default;

    GLuint getMatrixId() const final { return matrix_id_; }
    GLuint getProgramId() const final { return program_id_; }

private:

    void LoadShaders();
    void InitProgramId();
    void InitMatrixId();
    void CheckPrerequisites();

    std::string label_;

    GLuint matrix_id_{0};
    GLuint program_id_{0};

    Shader* fragment_shader_{nullptr};
    Shader* vertex_shader_{nullptr};
    ShaderFactory* shader_factory_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLPIPELINEPROGRAM_H_
