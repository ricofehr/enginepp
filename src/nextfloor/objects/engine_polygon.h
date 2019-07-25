/**
 *  @file engine_polygon.h
 *  @brief EnginePolygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINEPOLYGON_H_
#define NEXTFLOOR_OBJECTS_ENGINEPOLYGON_H_

#include <math.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_renderer.h"

#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace objects {

/**
 *  @class EnginePolygon
 *  @brief Interface which must be implemented to define any 3d graphic object.
 */
class EnginePolygon {

public:

    virtual ~EnginePolygon() = default;

    virtual void UpdateModelViewProjectionMatrix() = 0;
    virtual void Draw(nextfloor::objects::EngineRenderer* renderer) = 0;

    virtual void MoveLocation() noexcept = 0;
    virtual bool IsMoved() const = 0;

    virtual void set_distance(float distance) = 0;
    virtual void InverseMove() = 0;

    virtual float distance() const = 0;
    virtual glm::vec3 move() const = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 scale() const = 0;

protected:

    EnginePolygon() = default;

    EnginePolygon(EnginePolygon&&) = default;
    EnginePolygon& operator=(EnginePolygon&&) = default;
    EnginePolygon(const EnginePolygon&) = default;
    EnginePolygon& operator=(const EnginePolygon&) = default;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINEPOLYGON_H_
