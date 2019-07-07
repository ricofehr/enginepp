/**
 *  @file camera.cc
 *  @brief Camera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/camera.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

namespace {

/* Camera coords, used for collision computes */
static const std::vector<glm::vec3> sCameraCoords = {
    /* Back */
    {-0.25f,  0.5f,  0.25f},
    { 0.25f,  0.5f,  0.25f},
    { 0.25f, -0.5f,  0.25f},
    {-0.25f, -0.5f,  0.25f},
    /* Front */
    {-0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    {-0.25f, -0.5f, -0.25f},
    /* Left */
    {-0.25f,  0.5f,  0.25f},
    {-0.25f,  0.5f, -0.25f},
    {-0.25f, -0.5f, -0.25f},
    {-0.25f, -0.5f,  0.25f},
    /* Right */
    { 0.25f,  0.5f,  0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f,  0.25f},
    /* Top */
    {-0.25f,  0.5f,  0.25f},
    {-0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f,  0.25f},
    /* Roof */
    {-0.25f, -0.5f,  0.25f},
    {-0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f,  0.25f},
};


} // anonymous namespace

/* Define global active camera */
Camera* Camera::active_ = nullptr;

Camera::Camera()
       :Camera(glm::vec3(0.0f), 3.14f, 0.0f) {}

Camera::Camera(glm::vec3 location, float horizontal_angle, float vertical_angle)
{
    using nextfloor::core::CommonServices;

    type_ = kMODEL_CAMERA;
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(1.0f));
    horizontal_angle_ = horizontal_angle;
    vertical_angle_ = vertical_angle;
    fov_ = 45.0f;

    ComputeOrientation();

    if (active_ == nullptr) {
        set_active();
    }
}

void Camera::ComputeOrientation() noexcept
{
    /* Direction : Spherical coordinates to Cartesian coordinates conversion */
    direction_[0] = cos(vertical_angle_) * sin(horizontal_angle_);
    direction_[1] = sin(vertical_angle_);
    direction_[2] = cos(vertical_angle_) * cos(horizontal_angle_);

    /* Right vector */
    glm::vec3 right = glm::vec3(
        sin(horizontal_angle_ - 3.14f/2.0f),
        0,
        cos(horizontal_angle_ - 3.14f/2.0f)
    );

    /* head_ vector : perpendicular to both direction and right */
    head_ = glm::cross(right, direction_);
}

void Camera::ComputeFOV(float delta_fov) noexcept
{
    fov_ = fov_ + delta_fov;
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
}

} // namespace objects

} // namespace nextfloor
