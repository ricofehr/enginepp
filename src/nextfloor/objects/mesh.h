/**
 *  @file mesh.h
 *  @brief Mesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MESH_H_
#define NEXTFLOOR_OBJECTS_MESH_H_

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/objects/border.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Mesh
 *  @brief Interface who defines generic 3d object operations and shall be inherited by all other 3d models
 */
class Mesh {

public:

    virtual ~Mesh() = default;

    virtual void Draw() = 0;
    virtual std::vector<glm::vec3> getCoordsModelMatrixComputed() const noexcept = 0;

    virtual Mesh* add_child(std::unique_ptr<Mesh> child) noexcept = 0;
    virtual std::unique_ptr<Mesh> remove_child(Mesh* child) noexcept = 0;

    virtual bool IsLastObstacle(Mesh* obstacle) const noexcept = 0;
    virtual void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) noexcept = 0;

    virtual void PrepareDraw() = 0;

    virtual bool IsFrontPositionFilled() const noexcept = 0;
    virtual bool IsRightPositionFilled() const noexcept = 0;
    virtual bool IsLeftPositionFilled() const noexcept = 0;
    virtual bool IsBackPositionFilled() const noexcept = 0;
    virtual bool IsBottomPositionFilled() const noexcept = 0;
    virtual bool IsTopPositionFilled() const noexcept = 0;

    virtual int id() = 0;
    virtual glm::vec3 location() const noexcept = 0;
    virtual Border* border() const noexcept = 0;

    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual void set_parent(Mesh* parent) = 0;

    virtual bool ready() const = 0;
    virtual void toready() = 0;

    virtual bool IsCamera() const = 0;

protected:

    Mesh() = default;

    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_MESH_H_
