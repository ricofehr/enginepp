/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/gl_renderer_factory.h"

#include "nextfloor/renderer/gl_cube_renderer_engine.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

GlRendererFactory::GlRendererFactory()
{
    assert(!sInstanciated);
    sInstanciated = true;
}


nextfloor::objects::RendererEngine* GlRendererFactory::MakeCubeRenderer(std::string texture) noexcept
{
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<nextfloor::renderer::GlCubeRendererEngine>(texture);
    }
    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

GlRendererFactory::~GlRendererFactory()
{
    sInstanciated = false;
}

} // namespace factory

} // namespace nextfloor