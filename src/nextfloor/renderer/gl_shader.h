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
    virtual ~GlShader() override = default;

    virtual void LinkShader() override;
    virtual void DetachShader() override;
    virtual void CheckShader() override;
    virtual void CheckProgram() override;

    virtual GLuint shader_id() override { return shader_id_; }

protected:
    GlShader(std::string shader_filepath, GLuint program_id);

    GlShader(GlShader&&) = default;
    GlShader& operator=(GlShader&&) = default;
    GlShader(const GlShader&) = default;
    GlShader& operator=(const GlShader&) = default;


    std::string shader_filepath_;
    GLuint shader_id_;
    GLuint program_id_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSHADER_H_