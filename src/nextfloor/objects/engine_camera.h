/**
 *  @file engine_camera.h
 *  @brief EngineCamera class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINECAMERA_H_
#define NEXTFLOOR_OBJECTS_ENGINECAMERA_H_

#include "nextfloor/objects/engine_object.h"
#include "nextfloor/objects/model.h"

#include <glm/glm.hpp>
#include <vector>

namespace nextfloor {

namespace objects {

/**
 *  @class EngineCamera
 *  @brief Camera Abstract representation.\n
 */
class EngineCamera : public Model {

public:

    virtual ~EngineCamera() = default;

    virtual void ComputeOrientation() noexcept = 0;
    virtual void ComputeFOV(float delta_fov) noexcept = 0;

    virtual glm::vec3 direction() const = 0;
    virtual glm::vec3 head() const = 0;
    virtual float fov() const = 0;
    //virtual EngineCamera* active() = 0;

    virtual void set_active() = 0;

    static EngineCamera* active() { return active_; }

protected:

    EngineCamera() = default;

    EngineCamera(EngineCamera&&) = default;
    EngineCamera& operator=(EngineCamera&&) = default;
    EngineCamera(const EngineCamera&) = delete;
    EngineCamera& operator=(const EngineCamera&) = delete;

    /* At least one and only one current active Camera */
    static EngineCamera* active_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINECAMERA_H_

