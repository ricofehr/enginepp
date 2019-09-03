/**
 *  @file gl_renderer.cc
 *  @brief GlRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_renderer_engine.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {


GlRendererEngine::GlRendererEngine(std::string texture)
{
    using nextfloor::core::CommonServices;
    game_window_ = CommonServices::getFactory()->MakeSceneWindow();
    texture_ = texture;
}

}  // namespace renderer

}  // namespace nextfloor