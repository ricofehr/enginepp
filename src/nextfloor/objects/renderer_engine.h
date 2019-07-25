/**
 *  @file renderer_engine.h
 *  @brief Renderer Interface
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_RENDERERENGINE_H_
#define NEXTFLOOR_OBJECTS_RENDERERENGINE_H_

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

class RendererEngine {

public:

    virtual ~RendererEngine() = default;

    virtual void Draw(const glm::mat4& mvp) noexcept = 0;

protected:

    RendererEngine() = default;

    RendererEngine(RendererEngine&&) = default;
    RendererEngine& operator=(RendererEngine&&) = default;
    RendererEngine(const RendererEngine&) = default;
    RendererEngine& operator=(const RendererEngine&) = default;

};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_RENDERERENGINE_H_
