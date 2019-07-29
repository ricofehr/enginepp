/**
 *  @file shader.cc
 *  @brief Shader base class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/shader.h"

#include <vector>
#include <iostream>

#include "nextfloor/renderer/scene_window.h"

namespace nextfloor {

namespace renderer {

Shader::Shader(std::string shader_filepath)
{
    shader_filepath_ = shader_filepath;
};

void Shader::LinkShader()
{
    glAttachShader(SceneWindow::getProgramId(), shader_id_);
    glLinkProgram(SceneWindow::getProgramId());
}

void Shader::DetachShader()
{
    glDetachShader(SceneWindow::getProgramId(), shader_id_);
    glDeleteShader(shader_id_);
}

void Shader::CheckShader()
{
    /* Check Shader Compile */
    GLint result = GL_FALSE;
    int info_log_length;

    glGetShaderiv(shader_id_, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id_, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> shader_error_message(info_log_length + 1);
        glGetShaderInfoLog(shader_id_, info_log_length, nullptr, &shader_error_message[0]);
        std::cerr << &shader_error_message[0];
        exit(1);
    }
}

void Shader::CheckProgram()
{
    GLint result = GL_FALSE;
    int info_log_length;

    glGetProgramiv(SceneWindow::getProgramId(), GL_LINK_STATUS, &result);
    glGetProgramiv(SceneWindow::getProgramId(), GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(SceneWindow::getProgramId(), info_log_length, nullptr, &program_error_message[0]);
        std::cerr << &program_error_message[0];
        exit(1);
    }
}

} // namespace renderer

} // namespace nextfloor
