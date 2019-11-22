/**
 *  @file default_mesh.h
 *  @brief DefaultMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_DEFAULTMESH_H_
#define NEXTFLOOR_MESH_DEFAULTMESH_H_

#include "nextfloor/mesh/mesh.h"

#include <memory>
#include <glm/glm.hpp>

namespace nextfloor {

namespace mesh {

/**
 *  @class DefaultMesh
 *  @brief Abstract class who defines default implementations for some specialized functions of Mesh Interface
 */
class DefaultMesh : public Mesh {

public:
    ~DefaultMesh() override = default;

    bool IsCamera() const override { return false; }
    bool IsPlayer() const override { return false; }
    bool hasLayout() const override { return false; }

    /* Functions disabled by default, and overrided by dynamic Objects */
    std::vector<Mesh*> FindCollisionNeighbors() const override { return std::vector<Mesh*>(); }
    bool IsNeighborEligibleForCollision(const Mesh& neighbor) const override { return false; }
    void MoveLocation() override {}
    Mesh* UpdateChildPlacement(nextfloor::mesh::Mesh* child) override { return nullptr; }
    glm::vec3 movement() const override { return glm::vec3(0.0f); }
    float move_factor() const override { return 0.0f; }
    void set_move_factor(float move_factor) override {}
    void set_movement(const glm::vec3& movement) override {}

    /* Functions disabled by default, and overrided by ground Objects */
    std::vector<Mesh*> FindCollisionNeighborsOf(const Mesh& target) const override { return std::vector<Mesh*>(); }
    Mesh* AddIntoChild(std::unique_ptr<nextfloor::mesh::Mesh> mesh) override { return nullptr; }
    bool IsInside(const Mesh& mesh) const override { return false; }
    bool IsInside(const glm::vec3& location) const override { return false; }
    bool IsFrontPositionFilled() const override { return true; }
    bool IsRightPositionFilled() const override { return true; }
    bool IsBackPositionFilled() const override { return true; }
    bool IsLeftPositionFilled() const override { return true; }
    bool IsBottomPositionFilled() const override { return true; }
    bool IsTopPositionFilled() const override { return true; }

protected:
    DefaultMesh() = default;

    DefaultMesh(DefaultMesh&&) = delete;
    DefaultMesh& operator=(DefaultMesh&&) = delete;
    DefaultMesh(const DefaultMesh&) = delete;
    DefaultMesh& operator=(const DefaultMesh&) = delete;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_DEFAULTMESH_H_
