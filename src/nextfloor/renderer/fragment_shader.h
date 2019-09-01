/**
 *  @file fragment_shader.h
 *  @brief Fragment Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_
#define NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_

#include "nextfloor/renderer/shader.h"

#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  FragmentShader class, manage fragments for gl tunnel
 */
class FragmentShader : public Shader {

public:
    FragmentShader(std::string shader_path) : Shader(shader_path){};

    FragmentShader(FragmentShader&&) = default;
    FragmentShader& operator=(FragmentShader&&) = default;
    FragmentShader(const FragmentShader&) = default;
    FragmentShader& operator=(const FragmentShader&) = default;

    virtual ~FragmentShader() override = default;

    virtual void LoadShader() override final;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_