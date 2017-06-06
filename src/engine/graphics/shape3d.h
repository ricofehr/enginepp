/*
* Shape3D class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GRAPHICS_SHAPE3D_H_
#define ENGINE_GRAPHICS_SHAPE3D_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

namespace engine {
namespace graphics {

/* 3d object base class */
class Shape3D {

public:

    /* Default constructor, move and copy constructor / operator */
    Shape3D() = default;
    Shape3D(Shape3D&&) = default;
    Shape3D& operator=(Shape3D&&) = default;

    Shape3D(const Shape3D&) = default;
    Shape3D& operator=(const Shape3D&) = default;

    /* Abstract class, so virtual destructor */
    virtual ~Shape3D() = default;

    /* Thanks to active Camera and object moves, compute new gl coords */
    void ComputeMVP();
    /* Abstract function for Draw gl vertex */
    virtual void Draw() = 0;

    /* Apply Move Translate To Location */
    inline void MoveLocation() noexcept
    {
        if (distance_ != -1.0f) {
            location_ += move_ * distance_ * sMoveFactor;
            move_ = -move_;
        } else {
            location_ += move_ * sMoveFactor;
        }
        distance_ = -1.0f;
    }


    /* Accessors */
    bool IsMoved() const { return move_[0] != 0.0f || move_[1] != 0.0f || move_[2] != 0.0f; }
    bool IsMovedX() const { return move_[0] != 0.0f; }
    bool IsMovedY() const { return move_[1] != 0.0f; }
    bool IsMovedZ() const { return move_[2] != 0.0f; }
    float distance() const { return distance_; }
    glm::vec4 move() const { return move_ * sMoveFactor; }
    glm::vec4 location() const { return location_; }
    glm::vec3 scale() const { return scale_; }

    /* Mutators */
    void set_distance(float distance) { distance_ = distance; }
    void set_move(glm::vec4 move) { move_ = move; }
    void set_move(glm::vec3 move) { move_ = glm::vec4(move, 0.0f); }
    void InverseMove() { move_ = -move_; }

    /* Define a move factore who depends on fps */
    static float sMoveFactor;

protected:
    glm::mat4 mvp_;
    glm::vec4 location_;
    glm::vec4 move_;
    glm::vec3 scale_;
    float distance_;
};

}//namespace graphics
}//namespace engine

#endif //ENGINE_GRAPHICS_SHAPE3D_H_
