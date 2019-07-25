/**
 *  @file engine_renderer.h
 *  @brief Engine Renderer Interface
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINERENDERER_H_
#define NEXTFLOOR_OBJECTS_ENGINERENDERER_H_

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

class EngineRenderer {

public:

    virtual ~EngineRenderer() = default;

    virtual void Draw(const glm::mat4& mvp) noexcept = 0;

protected:

    EngineRenderer() = default;

    EngineRenderer(EngineRenderer&&) = default;
    EngineRenderer& operator=(EngineRenderer&&) = default;
    EngineRenderer(const EngineRenderer&) = default;
    EngineRenderer& operator=(const EngineRenderer&) = default;

};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_POLYGONS_ENGINERENDERER_H_
