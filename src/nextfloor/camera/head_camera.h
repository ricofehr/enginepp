/**
 *  @file head_camera.h
 *  @brief HeadCamera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_
#define NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_

#include "nextfloor/character/camera.h"

#include <glm/glm.hpp>

#include "nextfloor/character/character.h"

namespace nextfloor {

namespace camera {

/**
 *  @class HeadCamera
 *  @brief Concrete Implementation of Camera, for head vision
 */
class HeadCamera : public nextfloor::character::Camera {

public:
    HeadCamera(float horizontal_angle = kDefaultHorizontalAngle,
               float vertical_angle = kDefaultVerticalAngle,
               nextfloor::character::Character* owner = nullptr);
    ~HeadCamera() final = default;

    void ComputeOrientation() final;
    bool IsInFieldOfView(const nextfloor::mesh::Mesh& target) const final;
    glm::mat4 GetViewProjectionMatrix(float window_size_ratio) const final;

    glm::vec3 location() const final { return owner_->location(); }
    glm::vec3 direction() const final { return direction_; }
    glm::vec3 head() const final { return head_; }
    float fov() const final { return fov_; }

    void set_owner(nextfloor::character::Character* owner) final { owner_ = owner; }

    void increment_angles(float horizontal_angle, float vertical_angle) final
    {
        horizontal_angle_ += horizontal_angle;
        vertical_angle_ += vertical_angle;
    }

private:
    static constexpr float kDefaultHorizontalAngle = 180.0f;
    static constexpr float kDefaultVerticalAngle = 0.0f;

    static constexpr double kRadianByDegree = M_PI / 180.0f;
    static constexpr float kDefaultFov = 45.0f;
    static constexpr double kHalfPi = M_PI / 2.0f;

    static constexpr float kPerspectiveNear = 0.1f;
    static constexpr float kPerspectiveFar = 300.0f;

    glm::vec3 right_vector() const
    {
        return glm::vec3(sin(horizontal_angle_ - kHalfPi), 0, cos(horizontal_angle_ - kHalfPi));
    }

    float target_abs_angle(const nextfloor::mesh::Mesh& target) const
    {
        auto vector = target_vector(target);
        /* From dot formula : a . b = ||a|| ||b|| cos a^b */
        return abs(acos(glm::dot(direction(), vector) / (glm::length(direction()) * glm::length(vector))));
    }

    glm::vec3 target_vector(const nextfloor::mesh::Mesh& target) const { return target.location() - location(); }

    float increased_fov() const { return 4 * fov(); }

    glm::mat4 GetProjectionMatrix(float window_size_ratio) const;
    glm::mat4 GetViewMatrix() const;

    nextfloor::character::Character* owner_{nullptr};
    float horizontal_angle_;
    float vertical_angle_;
    float fov_;

    glm::vec3 direction_;
    glm::vec3 head_;
};

}  // namespace camera

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_HEADCAMERA_H_
