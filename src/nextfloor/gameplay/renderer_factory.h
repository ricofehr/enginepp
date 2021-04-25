/**
 *  @file renderer_factory.h
 *  @brief Abstract Factory Class for renderer objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_RENDERERFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_RENDERERFACTORY_H_

#include <memory>
#include <string>

#include "nextfloor/gameplay/renderer_engine.h"
#include "nextfloor/gameplay/scene_window.h"
#include "nextfloor/gameplay/scene_input.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class RendererFactory
 *  @brief Abstract Factory Pattern for flyweight renderer objects
 */
class RendererFactory {

public:
    virtual ~RendererFactory() = default;

    virtual RendererEngine* MakeCubeMapRenderer() = 0;
    virtual RendererEngine* MakeCubeRenderer(const std::string& texture) = 0;
    virtual SceneWindow* GetOrMakeSceneWindow() = 0;
    virtual std::unique_ptr<SceneInput> MakeSceneInput() = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_RENDERERFACTORY_H_
