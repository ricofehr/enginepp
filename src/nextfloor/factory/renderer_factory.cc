/**
 *  @file factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/renderer_factory.h"

#include "nextfloor/renderer/gl_cube_renderer.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

RendererFactory::RendererFactory()
{
    assert(!sInstanciated);
    sInstanciated = true;
}


nextfloor::objects::EngineRenderer* RendererFactory::MakeCubeRenderer(std::string texture) noexcept
{
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<nextfloor::renderer::GlCubeRenderer>(texture);
    }
    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

RendererFactory::~RendererFactory()
{
    sInstanciated = false;
}

} // namespace factory

} // namespace nextfloor