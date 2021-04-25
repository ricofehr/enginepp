/**
 *  @file gl_renderer.cc
 *  @brief GlRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/gl_renderer_engine.h"

namespace nextfloor {

namespace renderer {

GlRendererEngine::GlRendererEngine(PipelineProgram* pipeline_program)
{
    pipeline_program_ = pipeline_program;
}

}  // namespace renderer

}  // namespace nextfloor