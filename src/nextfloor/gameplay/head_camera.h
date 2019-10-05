/**
 *  @file head_camera.h
 *  @brief HeadCamera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_
#define NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_

#include "nextfloor/objects/camera.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class HeadCamera
 *  @brief HeadCamera, inherits Camera abstract class\n
 */
class HeadCamera : public nextfloor::objects::Camera {

public:
    HeadCamera(nextfloor::objects::Mesh* owner, float horizontal_angle, float vertical_angle);
    ~HeadCamera() final = default;

    void ComputeOrientation() final;
    void ComputeFOV(float delta_fov) final;
    bool IsInFieldOfView(const nextfloor::objects::Mesh& target) const final;

    glm::vec3 location() const final { return owner_->location(); }
    glm::vec3 direction() const final { return direction_; }
    glm::vec3 head() const final { return head_; }
    float fov() const final { return fov_; }
    void increment_angles(float horizontal_angle, float vertical_angle) final
    {
        horizontal_angle_ += horizontal_angle;
        vertical_angle_ += vertical_angle;
    }

private:
    nextfloor::objects::Mesh* owner_;
    float horizontal_angle_;
    float vertical_angle_;
    float fov_;

    glm::vec3 direction_;
    glm::vec3 head_;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_
