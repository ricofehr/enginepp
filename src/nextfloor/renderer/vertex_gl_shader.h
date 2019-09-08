/**
 *  @file vertex_gl_shader.h
 *  @brief Vertex GL Shader class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_VERTEXGLSHADER_H_
#define NEXTFLOOR_RENDERER_VERTEXGLSHADER_H_

#include "nextfloor/renderer/gl_shader.h"

#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  VertexShader class, manage vertex for gl tunnel
 */
class VertexGlShader : public GlShader {

public:
    VertexGlShader(std::string shader_path, GLuint program_id) : GlShader(shader_path, program_id){};

    VertexGlShader(VertexGlShader&&) = default;
    VertexGlShader& operator=(VertexGlShader&&) = default;
    VertexGlShader(const VertexGlShader&) = default;
    VertexGlShader& operator=(const VertexGlShader&) = default;

    virtual ~VertexGlShader() override = default;

    virtual void LoadShader() override final;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_VERTEXGLSHADER_H_