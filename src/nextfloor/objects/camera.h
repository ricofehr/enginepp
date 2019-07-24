/**
 *  @file camera.h
 *  @brief Camera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_CAMERA_H_
#define NEXTFLOOR_OBJECTS_CAMERA_H_

#include <glm/glm.hpp>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Camera
 *  @brief Camera model, inherits Model abstract class\n
 *  The active Camera is the eye of the GL Scene\n
 *  All 3d objects are displaying thanks to the projection matrix computed from camera position\n
 *  @see Polygon
 */
class Camera : public Model {

public:

    Camera();
    Camera(glm::vec3 location, float horizontal_angle, float vertical_angle);
    Camera(Camera&&) = default;
    Camera& operator=(Camera&&) = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;
    ~Camera() override = default;

    void ComputeOrientation() noexcept;
    void ComputeFOV(float delta_fov) noexcept;

    glm::vec3 direction() const { return direction_; }
    glm::vec3 head() const { return head_; }
    float fov() const { return fov_; }
    static Camera* active() { return active_; }

    void set_active() { active_ = this; }

private:

    /* At least one and only one current active Camera */
    static Camera* active_;

    float horizontal_angle_;
    float vertical_angle_;
    float fov_;

    glm::vec3 direction_;
    glm::vec3 head_;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_CAMERA_H_

