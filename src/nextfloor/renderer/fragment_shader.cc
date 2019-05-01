/**
 *  @file fragment_shader.cc
 *  @brief FragmentShader class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/fragment_shader.h"

#include <vector>

#include "nextfloor/core/file_io.h"

namespace nextfloor {

namespace renderer {

/**
 *   Compile and Load shader from file to ram
 */
void FragmentShader::LoadShader()
{
    nextfloor::core::FileIO file_io;

    /* Create the shader */
    shader_id_ = glCreateShader(GL_FRAGMENT_SHADER);

    /* Read the Vertex Shader code from the file */
    std::string shader_code = file_io.ReadFile(shader_filepath_);

    GLint result = GL_FALSE;
    int info_log_length;

    const char* shader_pointer = shader_code.c_str();
    glShaderSource(shader_id_, 1, &shader_pointer, nullptr);
    glCompileShader(shader_id_);

    /* Check Shader */
    glGetShaderiv(shader_id_, GL_COMPILE_STATUS, &result);
    glGetShaderiv(shader_id_, GL_INFO_LOG_LENGTH, &info_log_length);
    if (info_log_length > 0) {
        std::vector<char> fragmentshader_error_message(info_log_length + 1);
        glGetShaderInfoLog(shader_id_, info_log_length, nullptr, &fragmentshader_error_message[0]);
        std::cerr << &fragmentshader_error_message[0];
        exit(1);
    }
}

} // namespace renderer

} // namespace nextfloor
