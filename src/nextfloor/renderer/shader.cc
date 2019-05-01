/**
 *  @file shader.cc
 *  @brief Shader base class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/shader.h"

#include <vector>

namespace nextfloor {

namespace renderer {

GLuint Shader::sProgramId = -1;

Shader::Shader(std::string shader_filepath)
{
    shader_filepath_ = shader_filepath;

    /** Create GL Program */
    if (Shader::sProgramId == -1) {
        Shader::sProgramId = glCreateProgram();
    }
};

void Shader::LinkShader()
{
    /* Link the program */
    glAttachShader(Shader::sProgramId, shader_id_);
    glLinkProgram(Shader::sProgramId);
}

void Shader::DetachShader()
{
    glDetachShader(Shader::sProgramId, shader_id_);
    glDeleteShader(shader_id_);
}

void Shader::CheckProgram()
{
    GLint result = GL_FALSE;
    int info_log_length;

    /* Check the program */
    glGetProgramiv(Shader::sProgramId, GL_LINK_STATUS, &result);
    glGetProgramiv(Shader::sProgramId, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> program_error_message(info_log_length + 1);
        glGetProgramInfoLog(Shader::sProgramId, info_log_length, nullptr, &program_error_message[0]);
        std::cerr << &program_error_message[0];
        exit(1);
    }
}

} // namespace renderer

} // namespace nextfloor
