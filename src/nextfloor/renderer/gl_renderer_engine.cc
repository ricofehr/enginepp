/**
 *  @file gl_renderer.cc
 *  @brief GlRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_renderer_engine.h"

namespace nextfloor {

namespace renderer {

GlRendererEngine::GlRendererEngine(const std::string& texture, GLuint program_id, GLuint matrix_id)
{
    texture_ = texture;
    program_id_ = program_id;
    matrix_id_ = matrix_id;
}

}  // namespace renderer

}  // namespace nextfloor