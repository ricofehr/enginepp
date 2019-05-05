/**
 *  @file fragment_shader.cc
 *  @brief FragmentShader class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/fragment_shader.h"

#include <vector>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

void FragmentShader::LoadShader()
{
    using nextfloor::core::CommonServices;

    std::string shader_code = CommonServices::getFileIO().ReadFile(shader_filepath_);
    const char* shader_pointer = shader_code.c_str();

    shader_id_ = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader_id_, 1, &shader_pointer, nullptr);
    glCompileShader(shader_id_);

    CheckShader();
}

} // namespace renderer

} // namespace nextfloor
