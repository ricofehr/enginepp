/**
 *  @file gl_renderer_factory.h
 *  @brief GlRendererFactory Class for renderer flyweights objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_
#define NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_

#include "nextfloor/factory/renderer_factory.h"

#include <map>

#include "nextfloor/objects/renderer_engine.h"

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

    nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture) noexcept override;


private:
    std::map<std::string, std::unique_ptr<nextfloor::objects::RendererEngine>> renderers_;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_GLRENDERERFACTORY_H_
