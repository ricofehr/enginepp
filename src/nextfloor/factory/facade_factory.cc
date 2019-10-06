/**
 *  @file facade_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/facade_factory.h"

#include "nextfloor/factory/gl_renderer_factory.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace factory {

namespace {

static bool sInstanciated = false;

}

FacadeFactory::FacadeFactory()
{
    assert(!sInstanciated);
    renderer_factory_ = std::make_unique<GlRendererFactory>();
    sInstanciated = true;
}

nextfloor::gameplay::RendererEngine* FacadeFactory::MakeCubeRenderer(const std::string& texture)
{
    return renderer_factory_->MakeCubeRenderer(texture);
}

nextfloor::renderer::Shader* FacadeFactory::MakeVertexShader(const std::string& shader_path, unsigned int program_id)
{
    return renderer_factory_->MakeVertexShader(shader_path, program_id);
}

nextfloor::renderer::Shader* FacadeFactory::MakeFragmentShader(const std::string& shader_path, unsigned int program_id)
{
    return renderer_factory_->MakeFragmentShader(shader_path, program_id);
}

nextfloor::gameplay::SceneWindow* FacadeFactory::MakeSceneWindow()
{
    return renderer_factory_->MakeSceneWindow();
}

std::unique_ptr<nextfloor::renderer::SceneInput> FacadeFactory::MakeSceneInput()
{
    return renderer_factory_->MakeSceneInput();
}

FacadeFactory::~FacadeFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace factory

}  // namespace nextfloor