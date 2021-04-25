/**
 *  @file gl_scene_window.cc
 *  @brief GlPipelineProgram class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_pipeline_program.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cassert>

namespace nextfloor {

namespace renderer {

GlPipelineProgram::GlPipelineProgram(std::string label, ShaderFactory* shader_factory)
{
    label_ = std::move(label);
    shader_factory_ = shader_factory;

    /**
     *  Subroutines Order is matters
     */
    InitProgramId();
    LoadShaders();
    InitMatrixId();
    CheckPrerequisites();
}

void GlPipelineProgram::InitProgramId()
{
    program_id_ = glCreateProgram();
}

void GlPipelineProgram::LoadShaders()
{
    assert(shader_factory_ != nullptr);
    vertex_shader_ = shader_factory_->MakeVertexShader(label_, program_id_);
    fragment_shader_ = shader_factory_->MakeFragmentShader(label_, program_id_);

    vertex_shader_->LoadShader();
    fragment_shader_->LoadShader();
    vertex_shader_->LinkShader();
    fragment_shader_->LinkShader();
    vertex_shader_->CheckProgram();
    fragment_shader_->CheckProgram();
    vertex_shader_->DetachShader();
    fragment_shader_->DetachShader();
}

void GlPipelineProgram::InitMatrixId()
{
    matrix_id_ = glGetUniformLocation(program_id_, "MVP");
}


void GlPipelineProgram::CheckPrerequisites()
{
    //assert(matrix_id_ != 0);
    assert(program_id_ != 0);
}

}  // namespace renderer

}  // namespace nextfloor
