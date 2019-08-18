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

    virtual ~MeshPolygon() = default;

    virtual void UpdateModelViewProjectionMatrix() override;

    virtual inline void MoveLocation() noexcept override
    {
        location_ += movement() * move_factor();

        if (move_factor_ <= 0.0f) {
            InverseMove();
        }

        move_factor_ = 1.0f;
    }

    virtual bool IsMoved() const override
    {
        return movement_[0] != 0.0f || movement_[1] != 0.0f || movement_[2] != 0.0f;
    }

    virtual float move_factor() const override { return fabs(move_factor_); }
    virtual glm::vec3 movement() const override;
    virtual glm::vec3 location() const override { return location_; }
    virtual glm::vec3 scale() const override { return scale_; }

    virtual void set_move_factor(float move_factor) override { move_factor_ = move_factor; }
    virtual void set_movement(glm::vec3 movement) override { movement_ = movement; }
    virtual void InverseMove() override { movement_ = -movement_; }

protected:

    MeshPolygon() = default;

    MeshPolygon(MeshPolygon&&) = default;
    MeshPolygon& operator=(MeshPolygon&&) = default;
    MeshPolygon(const MeshPolygon&) = default;
    MeshPolygon& operator=(const MeshPolygon&) = default;


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
};

} // namespace polygons

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_SHAPE3D_H_
