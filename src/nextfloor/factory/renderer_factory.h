/**
 *  @file renderer_factory.h
 *  @brief Abstract Factory Class for renderer objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_RENDERERFACTORY_H_
#define NEXTFLOOR_FACTORY_RENDERERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/renderer_engine.h"
#include "nextfloor/renderer/shader.h"
#include "nextfloor/renderer/scene_window.h"
#include "nextfloor/renderer/scene_input.h"

namespace nextfloor {

namespace factory {

/**
 *  @class RendererFactory
 *  @brief Abstract Factory Pattern for flyweight renderer objects
 */
class RendererFactory {

public:
    RendererFactory(RendererFactory&&) = default;
    RendererFactory& operator=(RendererFactory&&) = default;
    RendererFactory(const RendererFactory&) = default;
    RendererFactory& operator=(const RendererFactory&) = default;

    virtual ~RendererFactory() = default;

    virtual nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture) = 0;
    virtual nextfloor::renderer::Shader* MakeVertexShader(std::string shader_path, unsigned int program_id) = 0;
    virtual nextfloor::renderer::Shader* MakeFragmentShader(std::string shader_path, unsigned int program_id) = 0;
    virtual nextfloor::renderer::SceneWindow* MakeSceneWindow() = 0;
    virtual std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput() = 0;

protected:
    RendererFactory() = default;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_RENDERERFACTORY_H_
