/**
 *  @file camera.h
 *  @brief Camera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_CAMERA_H_
#define NEXTFLOOR_OBJECTS_CAMERA_H_

#include <glm/glm.hpp>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Camera
 *  @brief Camera 3d model, inherits Model3D abstract class\n
 *  The active Camera is the eye of the GL Scene\n
 *  All 3d objects are displaying thanks to the projection matrix computed from camera position\n
 *  This class records hid (mouse, keyboard) events
 *  @see Shape3D
 */
class Camera : public Model {

public:

    /**
     *  Constructor
     */
    Camera();

    Camera(glm::vec3 location, float horizontal_angle, float vertical_angle);

    Camera(Camera&&) = default;

    Camera& operator=(Camera&&) = default;

    Camera(const Camera&) = delete;

    Camera& operator=(const Camera&) = delete;

    ~Camera() override = default;

    /*
     *  Accessors
     */
    glm::vec3 direction() const { return direction_; }
    glm::vec3 head() const { return head_; }
    float fov() const { return fov_; }
    static Camera* active() { return active_; }

    /*
     *  Mutators
     */
    void set_active() { active_ = this; }

    /**
     *  Compute Camera Orientation
     */
    void ComputeOrientation() noexcept;

    /**
     *  Compute Camera FOV
     */
    void ComputeFOV(float delta_fov) noexcept;

private:

    /** Horizontal orientation angle */
    float horizontal_angle_;

    /** Vertical orientation angle */
    float vertical_angle_;

    /** Field of view */
    float fov_;

    /* At least one and only one current active Camera */
    static Camera* active_;

    /** View direction vector */
    glm::vec3 direction_;

    /** Head coords */
    glm::vec3 head_;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_CAMERA_H_

