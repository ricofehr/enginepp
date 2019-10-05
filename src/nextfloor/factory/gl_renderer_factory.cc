/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/gl_renderer_factory.h"

#include "nextfloor/renderer/cube_gl_renderer_engine.h"
#include "nextfloor/renderer/gl_scene_window.h"
#include "nextfloor/renderer/gl_scene_input.h"
#include "nextfloor/renderer/vertex_gl_shader.h"
#include "nextfloor/renderer/fragment_gl_shader.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GlRendererFactory::GlRendererFactory()
{
    assert(!sInstanciated);
    sInstanciated = true;
}


nextfloor::gameplay::RendererEngine* GlRendererFactory::MakeCubeRenderer(const std::string& texture)
{
    static tbb::mutex renderer_mutex_;

    // renderer_mutex_.lock();
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<nextfloor::renderer::CubeGlRendererEngine>(texture);
    }
    // renderer_mutex_.unlock();

    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeVertexShader(const std::string& shader_path, unsigned int program_id)
{
    static tbb::mutex vertex_mutex_;

    // vertex_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::VertexGlShader;
        shaders_[shader_path] = std::make_unique<VertexGlShader>(shader_path, program_id);
    }
    // vertex_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeFragmentShader(const std::string& shader_path, unsigned int program_id)
{
    static tbb::mutex fragment_mutex_;

    // fragment_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::FragmentGlShader;
        shaders_[shader_path] = std::make_unique<FragmentGlShader>(shader_path, program_id);
    }
    // fragment_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::gameplay::SceneWindow* GlRendererFactory::MakeSceneWindow()
{
    static tbb::mutex scene_mutex_;

    if (scene_window_ != nullptr) {
        return scene_window_.get();
    }

    scene_mutex_.lock();
    if (scene_window_ == nullptr) {
        using nextfloor::renderer::GlSceneWindow;
        scene_window_ = std::make_unique<GlSceneWindow>();
        using nextfloor::core::CommonServices;
        CommonServices::initWindowSettings(scene_window_.get());
    }
    scene_mutex_.unlock();

    return scene_window_.get();
}

std::unique_ptr<nextfloor::renderer::SceneInput> GlRendererFactory::MakeSceneInput()
{
    using nextfloor::renderer::GlSceneInput;
    return std::make_unique<GlSceneInput>(MakeSceneWindow());
}

GlRendererFactory::~GlRendererFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor