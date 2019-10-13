/**
 *  @file shape3d.cc
 *  @brief Shape3D class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/mesh_polygon.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace nextfloor {

namespace polygons {

glm::vec3 MeshPolygon::movement() const
{
    return movement_;
}

void MeshPolygon::UpdateModelViewProjectionMatrix(const glm::mat4& view_projection_matrix)
{
    mvp_ = view_projection_matrix * GetModelMatrix() * glm::scale(scale_);
}

glm::mat4 MeshPolygon::GetModelMatrix()
{
    return glm::translate(glm::mat4(1.0f), location_);
}

}  // namespace polygons

}  // namespace nextfloor
