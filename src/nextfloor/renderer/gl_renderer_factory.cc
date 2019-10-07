/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/renderer/gl_renderer_factory.h"

#include "nextfloor/renderer/cube_gl_renderer_engine.h"
#include "nextfloor/renderer/gl_scene_window.h"
#include "nextfloor/renderer/gl_scene_input.h"

#include "nextfloor/renderer/gl_shader_factory.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace renderer {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GlRendererFactory::GlRendererFactory()
{
    assert(!sInstanciated);
    shader_factory_ = std::make_unique<GlShaderFactory>();
    sInstanciated = true;
}


nextfloor::gameplay::RendererEngine* GlRendererFactory::MakeCubeRenderer(const std::string& texture)
{
    static tbb::mutex renderer_mutex_;

    // renderer_mutex_.lock();
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<CubeGlRendererEngine>(texture);
    }
    // renderer_mutex_.unlock();

    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

nextfloor::gameplay::SceneWindow* GlRendererFactory::MakeSceneWindow()
{
    static tbb::mutex scene_mutex_;

    if (scene_window_ != nullptr) {
        return scene_window_.get();
    }

    scene_mutex_.lock();
    if (scene_window_ == nullptr) {
        scene_window_ = std::make_unique<GlSceneWindow>(shader_factory_.get());
        using nextfloor::core::CommonServices;
        CommonServices::initWindowSettings(scene_window_.get());
    }
    scene_mutex_.unlock();

    return scene_window_.get();
}

std::unique_ptr<nextfloor::gameplay::SceneInput> GlRendererFactory::MakeSceneInput()
{
    return std::make_unique<GlSceneInput>(MakeSceneWindow());
}

GlRendererFactory::~GlRendererFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace renderer

}  // namespace nextfloor