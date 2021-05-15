/**
 *  @file drawing_mesh.h
 *  @brief DrawingMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_DRAWINGMESH_H_
#define NEXTFLOOR_MESH_DRAWINGMESH_H_

#include "nextfloor/mesh/placement_mesh.h"

#include <vector>
#include <memory>

#include "nextfloor/mesh/polygon.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class DrawingMesh
 *  @brief Abstract class who add drawning functions to a Mesh object
 */
class DrawingMesh : public Mesh {

public:
    ~DrawingMesh() override = default;

    std::vector<std::pair<glm::mat4, std::string>> GetModelViewProjectionsAndTextureToDraw() const override;
    void PrepareDraw(const glm::mat4& view_projection_matrix) override;

    std::string class_name() const override { return "DrawingMesh"; }

protected:
    DrawingMesh() = default;
    std::vector<std::unique_ptr<Polygon>> polygons_;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_DRAWINGMESH_H_
