/**
 *  @file mesh_polygon.h
 *  @brief MeshPolygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGON_MESHPOLYGON_H_
#define NEXTFLOOR_POLYGON_MESHPOLYGON_H_

#include "nextfloor/mesh/polygon.h"

#include <glm/glm.hpp>
#include <string>

namespace nextfloor {

namespace polygon {

/**
 *  @class MeshPolygon
 *  @brief Abstract class which define the mesh polygons part
 */
class MeshPolygon : public nextfloor::mesh::Polygon {

public:
    ~MeshPolygon() noexcept override = default;

    void UpdateModelViewProjectionMatrix(const glm::mat4& view_projection_matrix) final;

    inline void MoveLocation() final
    {
        UpdateLocation();
        UpdateMovement();
        ResetMoveFactors();
    }

    bool IsMoved() const final { return movement_.x != 0.0f || movement_.y != 0.0f || movement_.z != 0.0f; }

    glm::vec3 movement() const final;
    glm::vec3 location() const final { return location_; }
    glm::vec3 scale() const final { return scale_; }
    glm::mat4 mvp() const final { return mvp_; }
    std::string texture() const final { return texture_; }

    void set_distance_factor(float distance_factor) final { distance_factor_ = distance_factor; }
    void set_move_factor(glm::vec3 move_factor) final { move_factor_ = move_factor; }
    void set_movement(const glm::vec3& movement) final { movement_ = movement; }

protected:
    MeshPolygon() = default;

    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    std::string texture_;

    glm::vec3 location_{0.0f, 0.0f, 0.0f};
    glm::vec3 scale_{0.0f, 0.0f, 0.0f};
    glm::vec3 movement_{0.0f, 0.0f, 0.0f};

    /** MOve factor with collision shape (1 -> no collision detected) */
    float distance_factor_ = 1.0f;
    glm::vec3 move_factor_{1.0f, 1.0f, 1.0f};

private:
    glm::mat4 GetModelMatrix();

    inline void UpdateLocation()
    {
        location_ += movement_ * distance_factor_;
    }

    inline void UpdateMovement()
    {
        if (distance_factor_ != 0.0f) {
            movement_ *= move_factor_;
        }
    }

    inline void ResetMoveFactors()
    {
        distance_factor_ = 1.0f;
        move_factor_ = glm::vec3(1.0f);
    }
};

}  // namespace polygon

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGON_MESHPOLYGON_H_
