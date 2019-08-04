/**
 *  @file polygon.h
 *  @brief Polygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_POLYGON_H_
#define NEXTFLOOR_OBJECTS_POLYGON_H_

#include <math.h>
#include <glm/glm.hpp>

#include "nextfloor/objects/renderer_engine.h"

#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace objects {

/**
 *  @class Polygon
 *  @brief Interface which must be implemented to define any 3d graphic object.
 */
class Polygon {

public:

    virtual ~Polygon() = default;

    virtual void UpdateModelViewProjectionMatrix() = 0;
    virtual void Draw(nextfloor::objects::RendererEngine* renderer) = 0;

    virtual void MoveLocation() noexcept = 0;
    virtual bool IsMoved() const = 0;

    virtual void set_distance(float distance) = 0;
    virtual void InverseMove() = 0;

    virtual float distance() const = 0;
    virtual glm::vec3 move() const = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 scale() const = 0;

protected:

    Polygon() = default;

    Polygon(Polygon&&) = default;
    Polygon& operator=(Polygon&&) = default;
    Polygon(const Polygon&) = default;
    Polygon& operator=(const Polygon&) = default;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_POLYGON_H_