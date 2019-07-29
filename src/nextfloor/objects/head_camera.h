/**
 *  @file head_camera.h
 *  @brief HeadCamera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_HEADCAMERA_H_
#define NEXTFLOOR_OBJECTS_HEADCAMERA_H_

#include "nextfloor/objects/camera.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace objects {

/**
 *  @class HeadCamera
 *  @brief HeadCamera, inherits Camera abstract class\n
 */
class HeadCamera : public Camera {

public:

    HeadCamera(Mesh* owner, float horizontal_angle, float vertical_angle);

    HeadCamera(HeadCamera&&) = default;
    HeadCamera& operator=(HeadCamera&&) = default;
    HeadCamera(const HeadCamera&) = delete;
    HeadCamera& operator=(const HeadCamera&) = delete;

    virtual ~HeadCamera() override = default;

    virtual void ComputeOrientation() noexcept override;
    virtual void ComputeFOV(float delta_fov) noexcept override;

    virtual glm::vec3 location() const override { return owner_->location(); }
    virtual glm::vec3 direction() const override { return direction_; }
    virtual glm::vec3 head() const override { return head_; }
    virtual float fov() const override { return fov_; }
    //virtual Camera* active() override { return active_; }

    virtual void set_active() override { active_ = this; }

private:

    Mesh* owner_;
    float horizontal_angle_;
    float vertical_angle_;
    float fov_;

    glm::vec3 direction_;
    glm::vec3 head_;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_HEADCAMERA_H_

