/**
 *  @file polygon.h
 *  @brief Polygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_POLYGON_H_
#define NEXTFLOOR_OBJECTS_POLYGON_H_

#include <glm/glm.hpp>
#include <string>

#include "nextfloor/objects/camera.h"

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

    virtual void UpdateModelViewProjectionMatrix(const glm::mat4& view_projection_matrix) = 0;

    virtual void MoveLocation() = 0;
    virtual bool IsMoved() const = 0;

    virtual void set_move_factor(float move_factor) = 0;
    virtual void set_movement(const glm::vec3& move) = 0;
    virtual void InverseMove() = 0;

    virtual float move_factor() const = 0;
    virtual glm::vec3 movement() const = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 scale() const = 0;
    virtual glm::mat4 mvp() const = 0;
    virtual std::string texture() const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_POLYGON_H_
