/**
 *  @file camera.h
 *  @brief Camera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_UNIVERSE_CAMERA_H_
#define NEXTFLOOR_UNIVERSE_CAMERA_H_

#include <glm/glm.hpp>

#include "nextfloor/hid/hid.h"

#include "nextfloor/universe/model3d.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::universe
 *  @brief World elements
 */
namespace universe {

/**
 *  @namespace nextfloor::universe::dynamic
 *  @brief universe dynamic object
 */
namespace dynamic {

/**
 *  @class Camera
 *  @brief Camera 3d model, inherits Model3D abstract class\n
 *  The active Camera is the eye of the GL Scene\n
 *  All 3d objects are displaying thanks to the projection matrix computed from camera position\n
 *  This class records hid (mouse, keyboard) events
 *  @see Shape3D
 */
class Camera : public Model3D {

public:

    /**
     *  Constructor
     */
    Camera();

    /**
     *  Constructor
     *  @param cx,cy,cz are camera location coordinates
     *  @param vx,vy,vz are camera view coordinates
     *  @param hx,hy,hz are camera head coordinates
     */
    Camera(float cx, float cy, float cz,
           float vx, float vy, float vz,
           float hx, float hy, float hz);

    /**
     *  Default Move constructor
     */
    Camera(Camera&&) = default;

    /**
     *  Default Move assignment
     */
    Camera& operator=(Camera&&) = default;

    /**
     *  Copy constructor Deleted (Model3D Inherit)
     */
    Camera(const Camera&) = delete;

    /**
     *  Copy assignment Deleted (Model3D Inherit)
     */
    Camera& operator=(const Camera&) = delete;

    /**
     *  Default destructor
     */
    ~Camera() override = default;

    /*
     *  Accessors
     */
    float fov() const { return fov_; }
    static Camera* active() { return active_; }

    /*
     *  Mutators
     */
    void set_active() { active_ = this; }

    /**
     *  Compute Camera Orientation
     */
    void ComputeOrientation(nextfloor::hid::HIDPointer position) noexcept;

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
};

} // namespace dynamic

} // namespace universe

} // namespace nextfloor

#endif // NEXTFLOOR_UNIVERSE_CAMERA_H_

