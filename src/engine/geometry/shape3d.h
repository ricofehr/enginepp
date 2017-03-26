/*
* Shape3D class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_GEOMETRY_SHAPE3D_H_
#define ENGINE_GEOMETRY_SHAPE3D_H_

#include <GL/glew.h>
#include <glm/glm.hpp>

namespace engine {
/* Forward declaration of class Camera */
namespace universe {
    class Camera;
}//namespace universe

namespace geometry {

/* 3d object base class */
class Shape3D {

public:
    /* Default constructor, move and copy constructor / operator */
    Shape3D() = default;
    Shape3D(Shape3D&&) = default;
    Shape3D& operator=(Shape3D&&) = default;

    Shape3D(const Shape3D&) = default;
    Shape3D& operator=(const Shape3D&) = default;

    void ComputeMVP(engine::universe::Camera *cam);
    virtual void Draw() = 0;

    bool IsMoved() const { return move_[0] != 0.0f || move_[1] != 0.0f || move_[2] != 0.0f; }
    bool IsMovedX() const { return move_[0] != 0.0f; }
    bool IsMovedY() const { return move_[1] != 0.0f; }
    bool IsMovedZ() const { return move_[2] != 0.0f; }
    float distance() const { return distance_; }
    glm::vec4 move() const { return move_; }
    glm::vec4 location() const { return location_; }

    void set_distance(float distance) { distance_ = distance; }
    void set_move(glm::vec4 move) { move_ = move; }
    void set_move(glm::vec3 move) { move_ = glm::vec4(move, 0.0f); }
    void InverseMove() { move_ = -move_; }

    /* Abstract class, so vietual destructor */
    virtual ~Shape3D() = default;

protected:
    glm::vec4 location_;
    glm::vec4 move_;
    glm::vec3 scale_;
    glm::mat4 mvp_;
    float distance_;
};

}//namespace geometry
}//namespace engine

#endif //ENGINE_GEOMETRY_SHAPE3D_H_
