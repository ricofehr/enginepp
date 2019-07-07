/**
 *  @file shape3d.h
 *  @brief Shape3D class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_POLYGON_H_
#define NEXTFLOOR_POLYGONS_POLYGON_H_

#include "nextfloor/objects/engine_polygon.h"

#include <math.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace polygons {

/**
 *  @class Shape3D
 *  @brief Interface which must be implemented to define any 3d graphic object.
 */
class Polygon : public nextfloor::objects::EnginePolygon {

public:

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~Polygon() = default;

    virtual void UpdateModelViewProjectionMatrix() override;

    //virtual void Draw(nextfloor::objects::EngineRenderer* renderer) = 0;


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

    /**
     *  Test if current shape is moving in the X axis
     *  @return
     *      0 if object is not moving on X\n
     *      1 if moving forward on X\n
     *     -1 if moving backward on X\n
     */
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

    /**
     *  Test if current shape is moving in the Y axis
     *  @return
     *      0 if object is not moving on Y\n
     *      1 if moving forward on Y\n
     *     -1 if moving backward on Y\n
     */
    inline int IsMovedY() const {
        if (move_[1] == 0.0f) {
            return 0;
        }

        if (move_[1] < 0.0f) {
            return -1;
        }

        return 1;
    }

    /**
     *  Test if current shape is moving in the Z axis
     *  @return
     *      0 if object is not moving on Z\n
     *      1 if moving forward on Z\n
     *     -1 if moving backward on Z\n
     */
    inline int IsMovedZ() const {
        if (move_[2] == 0.0f) {
            return 0;
        }

        if (move_[2] < 0.0f) {
            return -1;
        }

        return 1;
    }

    /*
     *  Accessors
     */
    virtual float distance() const override { return fabs(distance_); }
    virtual glm::vec3 move() const override;

    /*
     *  Mutators
     */
    virtual void set_distance(float distance) override { distance_ = distance; }
    void set_move(glm::vec3 move) { move_ = move; }
    virtual void InverseMove() override { move_ = -move_; }


    virtual glm::vec3 location() const override { return location_; }
    virtual glm::vec3 scale() const override { return scale_; }

protected:

    Polygon() = default;

    Polygon(Polygon&&) = default;

    Polygon& operator=(Polygon&&) = default;

    Polygon(const Polygon&) = default;

    Polygon& operator=(const Polygon&) = default;


    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    /** Center point */
    glm::vec3 location_;

    /** Scale factor for base coords */
    glm::vec3 scale_;

        /** Translation Vector */
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
