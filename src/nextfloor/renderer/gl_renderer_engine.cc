/**
 *  @file gl_renderer.cc
 *  @brief GlRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_renderer_engine.h"

namespace nextfloor {

namespace renderer {

GlRendererEngine::GlRendererEngine(const std::string& texture)
{
    texture_ = texture;
}

}  // namespace renderer

}  // namespace nextfloor