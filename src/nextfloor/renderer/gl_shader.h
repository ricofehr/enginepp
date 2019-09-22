/**
 *  @file gl_shader.h
 *  @brief GlShader abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSHADER_H_
#define NEXTFLOOR_RENDERER_GLSHADER_H_

#include "nextfloor/renderer/shader.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  Shader Interface class, define design for all program shaders
 */
class GlShader : public Shader {

public:
    ~GlShader() override = default;

    void LinkShader() final;
    void DetachShader() final;
    void CheckShader() final;
    void CheckProgram() final;

    GLuint shader_id() final { return shader_id_; }

protected:
    GlShader(const std::string& shader_filepath, GLuint program_id);

    GlShader(GlShader&&) = default;
    GlShader& operator=(GlShader&&) = default;
    GlShader(const GlShader&) = delete;
    GlShader& operator=(const GlShader&) = delete;


    std::string shader_filepath_;
    GLuint shader_id_;
    GLuint program_id_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSHADER_H_