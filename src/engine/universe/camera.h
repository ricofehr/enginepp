/**
 *   Camera class header
 *   @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *   Camera 3d model, inherits Model3D abstract class
 *
 *   The active Camera is the eye of the GL Scene.
 *   All 3d objects are displaying thanks to the projection
 *   matrix (ModelViewProjection, see Shape3D class) computed from camera position.
 *   This class records hid (mouse, keyboard) events.
 */

#ifndef ENGINE_UNIVERSE_CAMERA_H_
#define ENGINE_UNIVERSE_CAMERA_H_

#include <glm/glm.hpp>

#include "engine/universe/model3d.h"

namespace engine {

namespace universe {

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
    glm::vec3 direction() const { return direction_; }
    glm::vec3 head() const { return head_; }
    float fov() const { return fov_; }
    static Camera* active() { return active_; }

    /*
     *  Mutators
     */
    void set_active() { active_ = this; }

    /**
     *  Record Mouse and Keyboard Events 
     */
    void RecordHID() noexcept override final;

private:

    /** VIew direction vector */
    glm::vec3 direction_;

    /** Head coords */
    glm::vec3 head_;

    /** Horizontal orientation angle */
    float horizontal_angle_;

    /** Vertical orientation angle */
    float vertical_angle_;

    /** Field of view */
    float fov_;

    /* At least one and only one current active Camera */
    static Camera* active_;
};

} // namespace universe

} // namespace engine

#endif // ENGINE_UNIVERSE_CAMERA_H_

