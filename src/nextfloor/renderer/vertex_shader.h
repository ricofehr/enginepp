/**
 *  @file vertex_shader.h
 *  @brief Vertex Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_VERTEXSHADER_H_
#define NEXTFLOOR_RENDERER_VERTEXSHADER_H_

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
 *  VertexShader class, manage fragments for gl tunnel
 */
class VertexShader : public Shader {

public:

    /**
     *  Construtor
     *  @param shader_filename is the shader relative path
     */
    VertexShader(std::string shader_path) : Shader(shader_path) {};

    /**
     *  Default Move constructor
     */
    VertexShader(VertexShader&&) = default;

    /**
     *  Default Move assignment
     */
    VertexShader& operator=(VertexShader&&) = default;

    /**
     *  Default Copy constructor
     */
    VertexShader(const VertexShader&) = default;

    /**
     *  Default Copy assignment
     */
    VertexShader& operator=(const VertexShader&) = default;

    /**
     *  Default destructor
     */
    ~VertexShader() = default;

    /**
     *  Load the Shader
     */
    void LoadShader() override final;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_VERTEXSHADER_H_