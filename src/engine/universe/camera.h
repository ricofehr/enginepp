/*
* Camera class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_UNIVERSE_CAMERA_H_
#define ENGINE_UNIVERSE_CAMERA_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {
namespace universe {

/* Forward declaration of class Room */
class Room;

/* Manage 3d Camera */
class Camera : public Model3D {

public:
    Camera();
    Camera(float cx, float cy, float cz,
           float vx, float vy, float vz,
           float hx, float hy, float hz);

    /* Default move and copy constructor / operator */
    Camera(Camera&&) = default;
    Camera& operator=(Camera&&) = default;

    Camera(const Camera&) = default;
    Camera& operator=(const Camera&) = default;

    /* Default destructor */
    ~Camera() override = default;

    glm::vec3 direction() const { return direction_; }
    glm::vec3 head() const { return head_; }
    glm::vec3 location() const { return glm::vec3(border_.location()); }
    float fov() const { return fov_; }

    void Move();
    bool IsInRoom (const Room &room) const;

private:
    glm::vec3 direction_;
    glm::vec3 head_;
    float horizontal_angle_;
    float vertical_angle_;
    float fov_;
};

}//namespace universe
}//namespace engine

#endif //ENGINE_UNIVERSE_CAMERA_H_

