/**
 *  @file fragment_gl_shader.h
 *  @brief Fragment Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_FRAGMENTGLSHADER_H_
#define NEXTFLOOR_RENDERER_FRAGMENTGLSHADER_H_

#include "nextfloor/renderer/gl_shader.h"

#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  FragmentShader class, manage fragments for gl tunnel
 */
class FragmentGlShader : public GlShader {

public:
    FragmentGlShader(const std::string& shader_path, GLuint program_id) : GlShader(shader_path, program_id){};
    ~FragmentGlShader() final = default;

    void LoadShader() final;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_FRAGMENTGLSHADER_H_