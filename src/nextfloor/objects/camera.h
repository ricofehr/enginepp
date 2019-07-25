/**
 *  @file camera.h
 *  @brief Camera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_CAMERA_H_
#define NEXTFLOOR_OBJECTS_CAMERA_H_

#include <glm/glm.hpp>

#include "nextfloor/objects/engine_camera.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Camera
 *  @brief Camera model, inherits Model abstract class\n
 *  The active Camera is the eye of the GL Scene\n
 *  All 3d objects are displaying thanks to the projection matrix computed from camera position\n
 *  @see Polygon
 */
class Camera : public EngineCamera {

public:

    Camera();
    Camera(glm::vec3 location, float horizontal_angle, float vertical_angle);

    Camera(Camera&&) = default;
    Camera& operator=(Camera&&) = default;
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    virtual ~Camera() override = default;

    virtual void ComputeOrientation() noexcept override;
    virtual void ComputeFOV(float delta_fov) noexcept override;

    virtual glm::vec3 direction() const override { return direction_; }
    virtual glm::vec3 head() const override { return head_; }
    virtual float fov() const override { return fov_; }
    //virtual Camera* active() override { return active_; }

    virtual void set_active() override { active_ = this; }

private:

    float horizontal_angle_;
    float vertical_angle_;
    float fov_;

    glm::vec3 direction_;
    glm::vec3 head_;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_CAMERA_H_

