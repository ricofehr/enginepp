/**
 *  @file camera.h
 *  @brief Camera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_CAMERA_H_
#define NEXTFLOOR_OBJECTS_CAMERA_H_

#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

class Mesh;

/**
 *  @class Camera
 *  @brief Camera Abstract representation.\n
 */
class Camera {

public:
    virtual ~Camera() = default;

    virtual void ComputeOrientation() = 0;
    virtual void ComputeFOV(float delta_fov) = 0;
    virtual bool IsInFieldOfView(const Mesh& target) const = 0;

    virtual glm::vec3 location() const = 0;
    virtual glm::vec3 direction() const = 0;
    virtual glm::vec3 head() const = 0;
    virtual float fov() const = 0;

    virtual void set_owner(Mesh* owner) = 0;

    virtual void increment_angles(float horizontal_angle, float vertical_angle) = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_CAMERA_H_
