/**
 *  @file placement_mesh.h
 *  @brief PlacementMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_PLACEMENTMESH_H_
#define NEXTFLOOR_MESH_PLACEMENTMESH_H_

#include "nextfloor/mesh/composite_mesh.h"

#include <memory>
#include <vector>

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class PlacementMesh
 *  @brief Abstract class who add Border and placement box to a mesh
 */
class PlacementMesh : public CompositeMesh {

public:
    ~PlacementMesh() override = default;

    std::vector<Mesh*> GetMovingObjects() override;
    std::unique_ptr<Mesh> remove_child(Mesh* child) override;

    std::string class_name() const override { return "PlacementMesh"; }

protected:
    PlacementMesh() = default;

};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_PLACEMENTMESH_H_
