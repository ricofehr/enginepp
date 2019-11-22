/**
 *  @file mesh_polygon.h
 *  @brief MeshPolygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGON_MESHPOLYGON_H_
#define NEXTFLOOR_POLYGON_MESHPOLYGON_H_

#include "nextfloor/mesh/polygon.h"

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
        location_ += movement() * move_factor();

        if (move_factor_ <= 0.0f) {
            InverseMove();
        }

        move_factor_ = 1.0f;
    }

    bool IsMoved() const final { return movement_[0] != 0.0f || movement_[1] != 0.0f || movement_[2] != 0.0f; }

    glm::vec3 movement() const final;
    float move_factor() const final { return fabs(move_factor_); }
    glm::vec3 location() const final { return location_; }
    glm::vec3 scale() const final { return scale_; }
    glm::mat4 mvp() const final { return mvp_; }
    std::string texture() const final { return texture_; }

    void set_move_factor(float move_factor) final { move_factor_ = move_factor; }
    void set_movement(const glm::vec3& movement) final { movement_ = movement; }
    void InverseMove() final { movement_ = -movement_; }

protected:
    MeshPolygon() = default;

    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    std::string texture_;

    glm::vec3 location_{0.0f, 0.0f, 0.0f};
    glm::vec3 scale_{0.0f, 0.0f, 0.0f};
    glm::vec3 movement_{0.0f, 0.0f, 0.0f};

    /** MOve factor with collision shape (1 -> no collision detected) */
    float move_factor_{1.0f};

private:
    glm::mat4 GetModelMatrix();
};

}  // namespace polygon

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGON_MESHPOLYGON_H_
