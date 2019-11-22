/**
 *  @file collision_mesh.h
 *  @brief CollisionMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_COLLISIONMESH_H_
#define NEXTFLOOR_MESH_COLLISIONMESH_H_

#include "nextfloor/mesh/composite_mesh.h"

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/mesh/border.h"
#include "nextfloor/mesh/grid_box.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class CollisionMesh
 *  @brief Abstract class who add Border and placement box to a mesh
 */
class CollisionMesh : public CompositeMesh {

public:
    ~CollisionMesh() override = default;

    bool IsLastObstacle(Mesh* obstacle) const final;
    void UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) final;
    std::vector<Mesh*> GetMovingObjects() override;

    glm::vec3 location() const final { return border_->location(); }
    glm::vec3 dimension() const final { return border_->dimension(); }
    float diagonal() const final { return border_->diagonal(); }

    void set_gridcoords(std::vector<GridBox*> coords_list) final { coords_list_ = coords_list; }
    void delete_gridcoord(GridBox* grid_box) final;

    Border* border() const final { return border_.get(); }
    std::vector<glm::vec3> getCoordsModelMatrixComputed() const final;
    std::vector<glm::ivec3> coords() const final;
    void ClearCoords() final;
    std::unique_ptr<nextfloor::mesh::Mesh> remove_child(nextfloor::mesh::Mesh* child) override;

    std::string class_name() override { return "CollisionMesh"; }

protected:
    CollisionMesh() = default;

    bool IsMoved() const { return border_->IsMoved(); }
    void ResetObstacle();

    std::vector<GridBox*> coords_list_;
    std::unique_ptr<Border> border_{nullptr};

private:
    void InitCollisionEngine();
    void LogCollision(const Mesh& obstacle, float obstacle_distance);

    Mesh* obstacle_{nullptr};
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_COMPOSITEMESH_H_
