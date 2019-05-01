/**
 *  @file fragment_shader.h
 *  @brief Fragment Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_
#define NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_

#include "nextfloor/renderer/shader.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace renderer
 *  @brief rederer namespace
 */
namespace renderer {

/**
 *  FragmentShader class, manage fragments for gl tunnel
 */
class FragmentShader : public Shader {

public:

    /**
     *  Construtor
     *  @param shader_filepath is the shader relative path
     */
    FragmentShader(std::string shader_path) : Shader(shader_path) {};

    /**
     *  Default Move constructor
     */
    FragmentShader(FragmentShader&&) = default;

    /**
     *  Default Move assignment
     */
    FragmentShader& operator=(FragmentShader&&) = default;

    /**
     *  Default Copy constructor
     */
    FragmentShader(const FragmentShader&) = default;

    /**
     *  Default Copy assignment
     */
    FragmentShader& operator=(const FragmentShader&) = default;

    /**
     *  Default destructor
     */
    ~FragmentShader() = default;

    /**
     *  Load the Shader
     */
    void LoadShader() override final;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_FRAGMENTSHADER_H_