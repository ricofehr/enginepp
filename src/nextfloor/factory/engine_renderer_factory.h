/**
 *  @file engine_renderer_factory.h
 *  @brief Abstract Factory Class for renderer objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_ENGINERENDERERFACTORY_H_
#define NEXTFLOOR_FACTORY_ENGINERENDERERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_renderer.h"

namespace nextfloor {

namespace factory {

/**
 *  @class EngineRendererFactory
 *  @brief Abstract Factory Pattern for flyweight renderer objects
 */
class EngineRendererFactory {

public:

    EngineRendererFactory(EngineRendererFactory&&) = default;

    EngineRendererFactory& operator=(EngineRendererFactory&&) = default;

    EngineRendererFactory(const EngineRendererFactory&) = default;

    EngineRendererFactory& operator=(const EngineRendererFactory&) = default;

    virtual ~EngineRendererFactory() = default;

    virtual nextfloor::objects::EngineRenderer* MakeCubeRenderer(std::string texture) noexcept = 0;


protected:

    EngineRendererFactory() = default;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_ENGINERENDERERFACTORY_H_
