/**
 *  @file camera.cc
 *  @brief Camera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/universe/dynamic/camera.h"

namespace nextfloor {

namespace universe {

namespace dynamic {

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
       :Camera(15.0f, 100.0f, 15.0f, 15.0f, 0.0f, 15.0f, 0.0f, 0.0f, 1.0f) {}

Camera::Camera(float cx, float cy, float cz,
               float vx, float vy, float vz,
               float hx, float hy, float hz)
{
    direction_ = glm::vec3(vx, vy, vz);
    head_ = glm::vec3(hx, hy, hz);
    horizontal_angle_ = 3.14f;
    vertical_angle_ = 0.0;
    fov_ = 45.0f;

    using nextfloor::graphics::Border;
    border_ = std::make_unique<Border>(1.0f,
                glm::vec4(cx, cy, cz, 1.0f),
                glm::vec4(0.0f),
                sCameraCoords);

    type_ = kMODEL3D_CAMERA;

    if (active_ == nullptr) {
        set_active();
    }
}


void Camera::ComputeOrientation(nextfloor::hid::HIDPointer position) noexcept
{
    /* Compute new orientation */
    horizontal_angle_ += position.horizontal_delta_angle;
    vertical_angle_   += position.vertical_delta_angle;

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

    /* Reset move translate */
    border_->set_move(glm::vec3(0.0f));
}

void Camera::ComputeFOV(float delta_fov) noexcept
{
    fov_ = fov_ + delta_fov;
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
}

} // namespace dynamic

} // namespace universe

} // namespace nextfloor
