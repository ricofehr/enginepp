/**
 *  @file polygon.h
 *  @brief Polygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_POLYGON_H_
#define NEXTFLOOR_POLYGONS_POLYGON_H_

#include <glm/glm.hpp>
#include <string>

/* TODO: comment reason to enable experimental */
#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace mesh {

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

    virtual void set_distance_factor(float distance_factor) = 0;
    virtual void set_move_factor(glm::vec3 move_factor) = 0;
    virtual void set_movement(const glm::vec3& move) = 0;

    virtual glm::vec3 movement() const = 0;
    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 scale() const = 0;
    virtual glm::mat4 mvp() const = 0;
    virtual std::string texture() const = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_POLYGON_H_
