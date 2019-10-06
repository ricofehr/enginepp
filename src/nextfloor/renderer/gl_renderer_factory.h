/**
 *  @file gl_renderer_factory.h
 *  @brief GlRendererFactory Class for renderer flyweights objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLRENDERERFACTORY_H_
#define NEXTFLOOR_RENDERER_GLRENDERERFACTORY_H_

#include "nextfloor/gameplay/renderer_factory.h"

#include <map>
#include <tbb/mutex.h>

namespace nextfloor {

namespace renderer {

/**
 *  @class GlRendererFactory
 *  @brief GlRendererFactory Pattern for Universe Objects
 */
class GlRendererFactory : public nextfloor::gameplay::RendererFactory {

public:
    GlRendererFactory();
    ~GlRendererFactory() noexcept final;

    GlRendererFactory(GlRendererFactory&&) = default;
    GlRendererFactory& operator=(GlRendererFactory&&) = default;
    GlRendererFactory(const GlRendererFactory&) = delete;
    GlRendererFactory& operator=(const GlRendererFactory&) = delete;

    nextfloor::gameplay::RendererEngine* MakeCubeRenderer(const std::string& texture) final;
    Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id) final;
    Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id) final;
    nextfloor::gameplay::SceneWindow* MakeSceneWindow() final;
    std::unique_ptr<SceneInput> MakeSceneInput() final;

private:
    std::map<std::string, std::unique_ptr<nextfloor::gameplay::RendererEngine>> renderers_;
    std::map<std::string, std::unique_ptr<Shader>> shaders_;
    std::unique_ptr<nextfloor::gameplay::SceneWindow> scene_window_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLRENDERERFACTORY_H_
