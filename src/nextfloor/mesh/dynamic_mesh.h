/**
 *  @file dynamic_mesh.h
 *  @brief DynamicMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_DYNAMICMESH_H_
#define NEXTFLOOR_MESH_DYNAMICMESH_H_

#include "nextfloor/mesh/drawing_mesh.h"

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class ModelMesh
 *  @brief Abstract class who defines dynamic part for meshes
 */
class DynamicMesh : public DrawingMesh {

public:
    ~DynamicMesh() override = default;

    std::vector<Mesh*> FindCollisionNeighbors() const final;
    bool IsNeighborEligibleForCollision(const Mesh& neighbor) const final;
    void MoveLocation() override;

    glm::vec3 movement() const final { return border_->movement(); }
    float move_factor() const final;

    void set_move_factor(float move_factor) final;
    void set_movement(const glm::vec3& movement) final;

    std::string class_name() override { return "DynamicMesh"; }

protected:
    DynamicMesh() = default;

    DynamicMesh(DynamicMesh&&) = delete;
    DynamicMesh& operator=(DynamicMesh&&) = delete;
    DynamicMesh(const DynamicMesh&) = delete;
    DynamicMesh& operator=(const DynamicMesh&) = delete;

    bool IsMoved() const { return border_->IsMoved(); }

private:
    bool IsNeighborReachable(const Mesh& neighbor) const;
    bool IsInDirection(const Mesh& target) const;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_DYNAMICMESH_H_
