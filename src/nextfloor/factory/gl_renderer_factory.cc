/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/gl_renderer_factory.h"

#include "nextfloor/renderer/gl_cube_renderer_engine.h"
#include "nextfloor/renderer/game_window.h"
#include "nextfloor/renderer/vertex_shader.h"
#include "nextfloor/renderer/fragment_shader.h"

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


nextfloor::objects::RendererEngine* GlRendererFactory::MakeCubeRenderer(std::string texture)
{
    static tbb::mutex renderer_mutex_;

    renderer_mutex_.lock();
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<nextfloor::renderer::GlCubeRendererEngine>(texture);
    }
    renderer_mutex_.unlock();

    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeVertexShader(std::string shader_path)
{
    static tbb::mutex vertex_mutex_;

    vertex_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::VertexShader;
        shaders_[shader_path] = std::make_unique<VertexShader>(shader_path);
    }
    vertex_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeFragmentShader(std::string shader_path)
{
    static tbb::mutex fragment_mutex_;

    fragment_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::FragmentShader;
        shaders_[shader_path] = std::make_unique<FragmentShader>(shader_path);
    }
    fragment_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::renderer::SceneWindow* GlRendererFactory::MakeSceneWindow()
{
    static tbb::mutex scene_mutex_;

    scene_mutex_.lock();
    if (scene_window_ == nullptr) {
        using nextfloor::renderer::GameWindow;
        scene_window_ = std::make_unique<GameWindow>();
    }
    scene_mutex_.unlock();

    return scene_window_.get();
}

GlRendererFactory::~GlRendererFactory()
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor