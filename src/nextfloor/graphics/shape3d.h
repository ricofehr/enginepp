/**
 *  @file shape3d.h
 *  @brief Shape3D class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRAPHICS_SHAPE3D_H_
#define NEXTFLOOR_GRAPHICS_SHAPE3D_H_

#include <math.h>
#include <GL/glew.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL

namespace nextfloor {

namespace graphics {

/**
 *  @class Shape3D
 *  @brief Interface which must be implemented to define any 3d graphic object.
 */
class Shape3D {

public:

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~Shape3D() = default;

    void UpdateModelViewProjectionMatrix();

    virtual void Draw() = 0;

    inline void MoveLocation() noexcept
    {
        location_ += move() * distance();

        if (distance_ <= 0.0f) {
            InverseMove();
        }

        distance_ = 1.0f;
    }

    bool IsMoved() const 
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
    float distance() const { return fabs(distance_); }
    glm::vec4 location() const { return location_; }
    glm::vec3 scale() const { return scale_; }
    glm::vec4 move() const;

    /*
     *  Mutators
     */
    void set_distance(float distance) { distance_ = distance; }
    void set_move(glm::vec4 move) { move_ = move; }
    void set_move(glm::vec3 move) { move_ = glm::vec4(move, 0.0f); }
    void InverseMove() { move_ = -move_; }

protected:

    Shape3D() = default;

    Shape3D(Shape3D&&) = default;

    Shape3D& operator=(Shape3D&&) = default;

    Shape3D(const Shape3D&) = default;

    Shape3D& operator=(const Shape3D&) = default;


    /** MVP (Model View Projection) GL matrix for current 3d shape */
    glm::mat4 mvp_;

    /** Center point */
    glm::vec4 location_;

    /** Translation Vector */
    glm::vec4 move_;

    /** Scale factor for base coords */
    glm::vec3 scale_;

    /** Distance with collision shape (-1 -> no collision detected) */
    float distance_{-1.0f};

private:

    glm::mat4 GetProjectionMatrix();
    glm::mat4 GetViewMatrix();
    glm::mat4 GetModelMatrix();
};

} // namespace graphics

} // namespace nextfloor

#endif // NEXTFLOOR_GRAPHICS_SHAPE3D_H_
