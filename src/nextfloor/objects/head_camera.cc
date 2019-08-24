/**
 *  @file head_camera.cc
 *  @brief HeadCamera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/head_camera.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Camera* Camera::active_ = nullptr;

HeadCamera::HeadCamera(Mesh* owner, float horizontal_angle, float vertical_angle)
{
    owner_ = owner;
    horizontal_angle_ = horizontal_angle;
    vertical_angle_ = vertical_angle;
    fov_ = 45.0f;

    ComputeOrientation();

    if (active_ == nullptr) {
        set_active();
    }
}

void HeadCamera::ComputeOrientation() noexcept
{
    /* Direction : Spherical coordinates to Cartesian coordinates conversion */
    direction_[0] = cos(vertical_angle_) * sin(horizontal_angle_);
    direction_[1] = sin(vertical_angle_);
    direction_[2] = cos(vertical_angle_) * cos(horizontal_angle_);

    /* Right vector */
    glm::vec3 right = glm::vec3(sin(horizontal_angle_ - 3.14f / 2.0f), 0, cos(horizontal_angle_ - 3.14f / 2.0f));

    /* head_ vector : perpendicular to both direction and right */
    head_ = glm::cross(right, direction_);
}

void HeadCamera::ComputeFOV(float delta_fov) noexcept
{
    fov_ = fov_ + delta_fov;
    /* fov cant be less than 5° and more than 130° */
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
}

}  // namespace objects

}  // namespace nextfloor
