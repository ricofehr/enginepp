/**
 *  @file ground.h
 *  @brief Ground class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_GROUND_H_
#define NEXTFLOOR_PLAYGROUND_GROUND_H_

#include "nextfloor/mesh/placement_mesh.h"

#include <memory>

#include "nextfloor/playground/grid.h"

namespace nextfloor {

namespace playground {

/**
 *  @class Ground
 *  @brief Ground : define Abstract Ground class
 */
class Ground : public nextfloor::mesh::PlacementMesh {

public:
    ~Ground() override = default;

    std::vector<nextfloor::mesh::Mesh*> FindCollisionNeighborsOf(const nextfloor::mesh::Mesh& target) const final;
    void AddMeshToGrid(nextfloor::mesh::Mesh* object);
    void RemoveMeshToGrid(nextfloor::mesh::Mesh* object);
    void UpdateChildPlacementInGrid(nextfloor::mesh::Mesh* object);

    nextfloor::mesh::Mesh* add_child(std::unique_ptr<nextfloor::mesh::Mesh> object) final;
    std::unique_ptr<nextfloor::mesh::Mesh> remove_child(nextfloor::mesh::Mesh* child) final;

    nextfloor::mesh::Mesh* UpdateChildPlacement(nextfloor::mesh::Mesh* child) final;
    nextfloor::mesh::Mesh* TransfertChildToNeighbor(nextfloor::mesh::Mesh* child);
    nextfloor::mesh::Mesh* AddIntoChild(std::unique_ptr<nextfloor::mesh::Mesh> mesh) final;

    bool IsInside(const nextfloor::mesh::Mesh& mesh) const final;
    bool IsInside(const glm::vec3& location) const final;

    bool IsFrontPositionFilled() const final;
    bool IsRightPositionFilled() const final;
    bool IsBackPositionFilled() const final;
    bool IsLeftPositionFilled() const final;
    bool IsBottomPositionFilled() const final;
    bool IsTopPositionFilled() const final;

    bool hasLayout() const final { return true; }

    Grid* grid() const { return grid_.get(); }

    std::string class_name() final { return "Ground"; }

protected:
    std::unique_ptr<Grid> grid_{nullptr};
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_WALL_H_
