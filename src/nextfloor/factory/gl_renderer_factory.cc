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
    if (renderers_.find(texture) == renderers_.end()) {
        renderers_[texture] = std::make_unique<nextfloor::renderer::GlCubeRendererEngine>(texture);
    }
    assert(renderers_.find(texture) != renderers_.end());

    return renderers_[texture].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeVertexShader(std::string shader_path)
{
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::VertexShader;
        shaders_[shader_path] = std::make_unique<VertexShader>(shader_path);
    }
    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::renderer::Shader* GlRendererFactory::MakeFragmentShader(std::string shader_path)
{
    if (shaders_.find(shader_path) == shaders_.end()) {
        using nextfloor::renderer::FragmentShader;
        shaders_[shader_path] = std::make_unique<FragmentShader>(shader_path);
    }
    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

nextfloor::renderer::SceneWindow* GlRendererFactory::MakeSceneWindow()
{
    if (scene_window_ == nullptr) {
        using nextfloor::renderer::GameWindow;
        scene_window_ = std::make_unique<GameWindow>();
    }

    return scene_window_.get();
}

GlRendererFactory::~GlRendererFactory()
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor