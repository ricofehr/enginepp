/**
 *  @file renderer_factory.h
 *  @brief RendererFactory Class for renderer flyweights objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_RENDERERFACTORY_H_
#define NEXTFLOOR_FACTORY_RENDERERFACTORY_H_

#include "nextfloor/factory/engine_renderer_factory.h"

#include <map>

#include "nextfloor/objects/engine_renderer.h"

namespace nextfloor {

namespace factory {

/**
 *  @class RendererFactory
 *  @brief RendererFactory Pattern for Universe Objects
 */
class RendererFactory : public EngineRendererFactory {

public:

    RendererFactory();

    RendererFactory(RendererFactory&&) = default;

    RendererFactory& operator=(RendererFactory&&) = default;

    RendererFactory(const RendererFactory&) = default;

    RendererFactory& operator=(const RendererFactory&) = default;

    virtual ~RendererFactory() override;

    nextfloor::objects::EngineRenderer* MakeCubeRenderer(std::string texture) noexcept override;


private:

    std::map<std::string, std::unique_ptr<nextfloor::objects::EngineRenderer>> renderers_;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_ENGINEFACTORY_H_
