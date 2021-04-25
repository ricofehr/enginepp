/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/renderer/gl_renderer_factory.h"

#include "nextfloor/renderer/cube_map_gl_renderer_engine.h"
#include "nextfloor/renderer/cube_gl_renderer_engine.h"
#include "nextfloor/renderer/gl_scene_window.h"
#include "nextfloor/renderer/gl_scene_input.h"
#include "nextfloor/renderer/gl_shader_factory.h"
#include "nextfloor/renderer/gl_pipeline_program.h"

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

nextfloor::gameplay::RendererEngine* GlRendererFactory::MakeCubeMapRenderer()
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    if (cube_map_renderer_ == nullptr) {
        if (pipeline_programs_.find(kCubeMapRendererLabel) == pipeline_programs_.end()) {
            pipeline_programs_[kCubeMapRendererLabel] = std::make_unique<GlPipelineProgram>(kCubeMapRendererLabel, shader_factory_.get());
        }
        cube_map_renderer_ = std::make_unique<CubeMapGlRendererEngine>(pipeline_programs_[kCubeMapRendererLabel].get());
    }

    assert(cube_map_renderer_ != nullptr);

    return cube_map_renderer_.get();
}

nextfloor::gameplay::RendererEngine* GlRendererFactory::MakeCubeRenderer(const std::string& texture)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    if (renderers_.find(texture) == renderers_.end()) {
        if (pipeline_programs_.find(kCubeRendererLabel) == pipeline_programs_.end()) {
            pipeline_programs_[kCubeRendererLabel] = std::make_unique<GlPipelineProgram>(kCubeRendererLabel, shader_factory_.get());
        }
        renderers_[texture] = std::make_unique<CubeGlRendererEngine>(texture,pipeline_programs_[kCubeRendererLabel].get());
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
        scene_window_ = std::make_unique<GlSceneWindow>();
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