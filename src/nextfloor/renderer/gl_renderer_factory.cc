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
    tbb::mutex::scoped_lock lock_map(mutex_);

    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<CubeGlRendererEngine>(
          texture, GetOrMakeSceneWindow()->getProgramId(), GetOrMakeSceneWindow()->getMatrixId());
    }

    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

nextfloor::gameplay::SceneWindow* GlRendererFactory::GetOrMakeSceneWindow()
{
    if (scene_window_ != nullptr) {
        return scene_window_.get();
    }

    tbb::mutex::scoped_lock lock_map(mutex_);

    if (scene_window_ == nullptr) {
        scene_window_ = std::make_unique<GlSceneWindow>(shader_factory_.get());
    }

    return scene_window_.get();
}

std::unique_ptr<nextfloor::gameplay::SceneInput> GlRendererFactory::MakeSceneInput()
{
    return std::make_unique<GlSceneInput>(static_cast<GLFWwindow*>(GetOrMakeSceneWindow()->window()));
}

GlRendererFactory::~GlRendererFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace renderer

}  // namespace nextfloor