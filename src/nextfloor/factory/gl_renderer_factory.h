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

    GlRendererFactory(GlRendererFactory&&) = default;
    GlRendererFactory& operator=(GlRendererFactory&&) = default;
    GlRendererFactory(const GlRendererFactory&) = default;
    GlRendererFactory& operator=(const GlRendererFactory&) = default;

    virtual ~GlRendererFactory() override;

    virtual nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture) override;
    virtual nextfloor::renderer::Shader* MakeVertexShader(std::string shader_path, unsigned int program_id) override;
    virtual nextfloor::renderer::Shader* MakeFragmentShader(std::string shader_path, unsigned int program_id) override;
    virtual nextfloor::renderer::SceneWindow* MakeSceneWindow() override;
    virtual std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput() override;

private:
    std::map<std::string, std::unique_ptr<nextfloor::objects::RendererEngine>> renderers_;
    std::map<std::string, std::unique_ptr<nextfloor::renderer::Shader>> shaders_;
    std::unique_ptr<nextfloor::renderer::SceneWindow> scene_window_;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_
