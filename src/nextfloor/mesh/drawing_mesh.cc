/**
 *  @file drawing_mesh.cc
 *  @brief DrawingMesh class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/mesh/drawing_mesh.h"

#include <tbb/tbb.h>

namespace nextfloor {

namespace mesh {

void DrawingMesh::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) { objects_[i]->PrepareDraw(view_projection_matrix); });

    tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) {
        polygons_[counter]->UpdateModelViewProjectionMatrix(view_projection_matrix);
    });
}

std::vector<std::pair<glm::mat4, std::string>> DrawingMesh::GetModelViewProjectionsAndTextureToDraw() const
{
    std::vector<std::pair<glm::mat4, std::string>> mvps_with_texture;
    for (auto& polygon : polygons_) {
        mvps_with_texture.push_back(std::pair<glm::mat4, std::string>(polygon->mvp(), polygon->texture()));
    }
    return mvps_with_texture;
}

}  // namespace mesh

}  // namespace nextfloor
