/**
 *  @file gl_shader.cc
 *  @brief GlShader base class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_shader.h"

#include <vector>
#include <iostream>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

GlShader::GlShader(std::string shader_filepath, GLuint program_id)
{
    shader_filepath_ = shader_filepath;
    program_id_ = program_id;
};

void GlShader::LinkShader()
{
    glAttachShader(program_id_, shader_id_);
    glLinkProgram(program_id_);
}

void GlShader::DetachShader()
{
    glDetachShader(program_id_, shader_id_);
    glDeleteShader(shader_id_);
}

void GlShader::CheckShader()
{
    /* Check Shader Compile */
    GLint result = GL_FALSE;
    int info_log_length;

    glGetShaderiv(shader_id_, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id_, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> shader_error_message(info_log_length + 1);
        glGetShaderInfoLog(shader_id_, info_log_length, nullptr, &shader_error_message[0]);

        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine(std::string(shader_error_message.begin(), shader_error_message.end()));
        CommonServices::getExit()->ExitOnError();
    }
}

void GlShader::CheckProgram()
{
    using nextfloor::core::CommonServices;

    GLint result = GL_FALSE;
    int info_log_length;

    glGetProgramiv(program_id_, GL_LINK_STATUS, &result);
    glGetProgramiv(program_id_, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(program_id_, info_log_length, nullptr, &program_error_message[0]);

        using nextfloor::core::CommonServices;
        CommonServices::getLog()->WriteLine(std::string(program_error_message.begin(), program_error_message.end()));
        CommonServices::getExit()->ExitOnError();
    }
}

}  // namespace renderer

}  // namespace nextfloor
