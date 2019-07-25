/**
 *  @file vertex_shader.h
 *  @brief Vertex Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_VERTEXSHADER_H_
#define NEXTFLOOR_RENDERER_VERTEXSHADER_H_

#include "nextfloor/renderer/shader.h"

#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  VertexShader class, manage vertex for gl tunnel
 */
class VertexShader : public Shader {

public:

    VertexShader(std::string shader_path) : Shader(shader_path) {};

    VertexShader(VertexShader&&) = default;
    VertexShader& operator=(VertexShader&&) = default;
    VertexShader(const VertexShader&) = default;
    VertexShader& operator=(const VertexShader&) = default;

    virtual ~VertexShader() = default;

    virtual void LoadShader() override final;
};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_VERTEXSHADER_H_