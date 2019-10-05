/**
 *  @file renderer_engine.h
 *  @brief Renderer Interface
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_RENDERERENGINE_H_
#define NEXTFLOOR_GAMEPLAY_RENDERERENGINE_H_

#include <glm/glm.hpp>

namespace nextfloor {

namespace gameplay {

class RendererEngine {

public:
    virtual ~RendererEngine() = default;

    virtual void Draw(const glm::mat4& mvp) = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_RENDERERENGINE_H_
