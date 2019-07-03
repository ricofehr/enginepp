/**
 *  @file mesh.h
 *  @brief Mesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINEOBJECT_H_
#define NEXTFLOOR_OBJECTS_ENGINEOBJECT_H_

#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace nextfloor {

namespace objects {

/**
 *  @class EngineObject
 *  @brief Interface who defines generic 3d object operations and shall be inherited by all other 3d models
 */
class EngineObject {

public:

    EngineObject(EngineObject&&) = default;

    EngineObject& operator=(EngineObject&&) = default;

    EngineObject(const EngineObject&) = delete;

    EngineObject& operator=(const EngineObject&) = delete;

    virtual ~EngineObject() = default;

    virtual void Draw() = 0;

    virtual EngineObject* add_child(std::unique_ptr<EngineObject> child) noexcept = 0;

    virtual std::unique_ptr<EngineObject> remove_child(EngineObject* child) noexcept = 0;

    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const noexcept = 0;

    virtual void lock() = 0;

    virtual void unlock() = 0;

    virtual void set_parent(EngineObject* parent) = 0;

    virtual bool IsUniverse() const = 0;
    virtual bool IsRoom() const = 0;
    virtual bool IsWall() const = 0;
    virtual bool IsCamera() const = 0;

    virtual int id() = 0;

    virtual glm::vec3 location() const noexcept = 0;

protected:

    EngineObject() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINEOBJECT_H_
