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
        location_ += move() * distance();

        if (distance_ <= 0.0f) {
            InverseMove();
        }

        distance_ = 1.0f;
    }

    virtual bool IsMoved() const override
    {
        return move_[0] != 0.0f || move_[1] != 0.0f || move_[2] != 0.0f;
    }

    inline int IsMovedX() const
    {
        if (move_[0] == 0.0f) {
            return 0;
        }

        if (move_[0] < 0.0f) {
            return -1;
        }

        return 1;
    }

    inline int IsMovedY() const {
        if (move_[1] == 0.0f) {
            return 0;
        }

        if (move_[1] < 0.0f) {
            return -1;
        }

        return 1;
    }

    inline int IsMovedZ() const {
        if (move_[2] == 0.0f) {
            return 0;
        }

        if (move_[2] < 0.0f) {
            return -1;
        }

        return 1;
    }

    virtual float distance() const override { return fabs(distance_); }
    virtual glm::vec3 move() const override;
    virtual glm::vec3 location() const override { return location_; }
    virtual glm::vec3 scale() const override { return scale_; }

    virtual void set_distance(float distance) override { distance_ = distance; }
    void set_move(glm::vec3 move) { move_ = move; }
    virtual void InverseMove() override { move_ = -move_; }

protected:

    MeshPolygon() = default;

    MeshPolygon(MeshPolygon&&) = default;
    MeshPolygon& operator=(MeshPolygon&&) = default;
    MeshPolygon(const MeshPolygon&) = default;
    MeshPolygon& operator=(const MeshPolygon&) = default;


    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    glm::vec3 location_;
    glm::vec3 scale_;
    glm::vec3 move_;

    /** Distance with collision shape (-1 -> no collision detected) */
    float distance_{-1.0f};

private:

    glm::mat4 GetProjectionMatrix();
    glm::mat4 GetViewMatrix();
    glm::mat4 GetModelMatrix();
};

} // namespace polygons

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_SHAPE3D_H_
