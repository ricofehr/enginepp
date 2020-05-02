/**
 *  @file head_camera.cc
 *  @brief HeadCamera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/camera/head_camera.h"

#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace nextfloor {

namespace camera {

HeadCamera::HeadCamera(float horizontal_angle, float vertical_angle, nextfloor::character::Character* owner)
{
    owner_ = owner;
    horizontal_angle_ = glm::radians(horizontal_angle);
    vertical_angle_ = glm::radians(vertical_angle);
    fov_ = glm::radians(kDefaultFov);

    ComputeOrientation();
}

void HeadCamera::ComputeOrientation()
{
    /* Direction : Spherical coordinates to Cartesian coordinates conversion */
    direction_[0] = cos(vertical_angle_) * sin(horizontal_angle_);
    direction_[1] = sin(vertical_angle_);
    direction_[2] = cos(vertical_angle_) * cos(horizontal_angle_);

    /* head_ vector : perpendicular to both direction and right vector */
    head_ = glm::cross(right_vector(), direction_);
}

bool HeadCamera::IsInFieldOfView(const nextfloor::mesh::Mesh& target) const
{
    /* For rooms, display always the one where we're in */
    if (target.IsInside(location())) {
        return true;
    }

    return target_abs_angle(target) <= fov();
}

glm::mat4 HeadCamera::GetViewProjectionMatrix(float window_size_ratio) const
{
    return GetProjectionMatrix(window_size_ratio) * GetViewMatrix();
}

glm::mat4 HeadCamera::GetProjectionMatrix(float window_size_ratio) const
{
    return glm::perspective(fov(), window_size_ratio, kPerspectiveNear, kPerspectiveFar);
}

glm::mat4 HeadCamera::GetViewMatrix() const
{
    return glm::lookAt(location(), location() + direction(), head());
}

glm::vec3 HeadCamera::owner_movement() const
{
    auto movement = direction();
    // if (!owner_->is_flying()) {
    //     movement.y = -1.0f;
    // }

    return movement;
}

}  // namespace camera

}  // namespace nextfloor
