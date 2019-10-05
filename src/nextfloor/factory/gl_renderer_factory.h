/**
 *  @file gl_renderer_factory.h
 *  @brief GlRendererFactory Class for renderer flyweights objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_
#define NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_

#include "nextfloor/factory/renderer_factory.h"

#include <map>
#include <tbb/mutex.h>

namespace nextfloor {

namespace factory {

/**
 *  @class GlRendererFactory
 *  @brief GlRendererFactory Pattern for Universe Objects
 */
class GlRendererFactory : public RendererFactory {

public:
    GlRendererFactory();
    ~GlRendererFactory() noexcept final;

    GlRendererFactory(GlRendererFactory&&) = default;
    GlRendererFactory& operator=(GlRendererFactory&&) = default;
    GlRendererFactory(const GlRendererFactory&) = delete;
    GlRendererFactory& operator=(const GlRendererFactory&) = delete;

    nextfloor::objects::RendererEngine* MakeCubeRenderer(const std::string& texture) final;
    nextfloor::renderer::Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id) final;
    nextfloor::renderer::Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id) final;
    nextfloor::gameplay::SceneWindow* MakeSceneWindow() final;
    std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput() final;

private:
    std::map<std::string, std::unique_ptr<nextfloor::objects::RendererEngine>> renderers_;
    std::map<std::string, std::unique_ptr<nextfloor::renderer::Shader>> shaders_;
    std::unique_ptr<nextfloor::gameplay::SceneWindow> scene_window_;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_
