/**
 *  @file head_camera.cc
 *  @brief HeadCamera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/head_camera.h"

#include <math.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

HeadCamera::HeadCamera(float horizontal_angle, float vertical_angle, nextfloor::objects::Mesh* owner)
{
    owner_ = owner;
    horizontal_angle_ = horizontal_angle;
    vertical_angle_ = vertical_angle;
    fov_ = 45.0f;

    ComputeOrientation();
}

void HeadCamera::ComputeOrientation()
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

void HeadCamera::ComputeFOV(float delta_fov)
{
    fov_ = fov_ + delta_fov;
    /* fov cant be less than 5° and more than 130° */
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
}

bool HeadCamera::IsInFieldOfView(const nextfloor::objects::Mesh& target) const
{
    using nextfloor::core::CommonServices;

    /* For rooms, display always the one where we're in */
    if (target.IsInside(location())) {
        return true;
    }

    /*
     * For nearer object, or bigger object, increase fov
     * TODO : avoid magics numbers
     */
    auto vector = target.location() - location();
    auto cos_camera_vector = glm::dot(direction(), vector) / (glm::length(direction()) * glm::length(vector));
    float camera_fov = glm::length(vector) < 4.0f || target.diagonal() > 3.0f ? 95.0f : fov();

    /* From dot product rule to known wich side is a point from an other */
    return cos_camera_vector >= cos(camera_fov * M_PI / 180.0f);
}

glm::mat4 HeadCamera::GetViewProjectionMatrix(float window_size_ratio) const
{
    return GetProjectionMatrix(window_size_ratio) * GetViewMatrix();
}

glm::mat4 HeadCamera::GetProjectionMatrix(float window_size_ratio) const
{
    return glm::perspective(glm::radians(fov()), window_size_ratio, 0.1f, 300.0f);
}

glm::mat4 HeadCamera::GetViewMatrix() const
{
    return glm::lookAt(location(), location() + direction(), head());
}


}  // namespace gameplay

}  // namespace nextfloor
