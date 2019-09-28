/**
 *  @file mesh_polygon.h
 *  @brief MeshPolygon class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_MESHPOLYGON_H_
#define NEXTFLOOR_POLYGONS_MESHPOLYGON_H_

#include "nextfloor/objects/polygon.h"

#include <math.h>
#include <glm/glm.hpp>

#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace polygons {

/**
 *  @class MeshPolygon
 *  @brief Abstract class which define the mesh polygons part
 */
class MeshPolygon : public nextfloor::objects::Polygon {

public:
    ~MeshPolygon() noexcept override = default;

    void UpdateModelViewProjectionMatrix() final;

    inline void MoveLocation() final
    {
        location_ += movement() * move_factor();

        if (move_factor_ <= 0.0f) {
            InverseMove();
        }

        move_factor_ = 1.0f;
    }

    bool IsMoved() const final { return movement_[0] != 0.0f || movement_[1] != 0.0f || movement_[2] != 0.0f; }

    float move_factor() const final { return fabs(move_factor_); }
    glm::vec3 movement() const final;
    glm::vec3 location() const final { return location_; }
    glm::vec3 scale() const final { return scale_; }

    void set_active_camera(nextfloor::objects::Camera* active_camera) final { active_camera_ = active_camera; }
    void set_move_factor(float move_factor) final { move_factor_ = move_factor; }
    void set_movement(const glm::vec3& movement) final { movement_ = movement; }
    void InverseMove() final { movement_ = -movement_; }

protected:
    MeshPolygon() = default;

    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    glm::vec3 location_{0.0f, 0.0f, 0.0f};
    glm::vec3 scale_{0.0f, 0.0f, 0.0f};
    glm::vec3 movement_{0.0f, 0.0f, 0.0f};

    /** MOve factor with collision shape (1 -> no collision detected) */
    float move_factor_{1.0f};

private:
    glm::mat4 GetProjectionMatrix();
    glm::mat4 GetViewMatrix();
    glm::mat4 GetModelMatrix();

    nextfloor::objects::Camera* active_camera_;
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRAPHICS_SHAPE3D_H_
