/**
 *  @file mesh_polygon.cc
 *  @brief MeshPolygon class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/mesh_polygon.h"

/* Need Experimental flag for transform methods */
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>


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
