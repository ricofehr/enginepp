/**
 *  @file cube.cc
 *  @brief Cube class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace polygons {

Cube::Cube(const glm::vec3& location, float scale) : Cube(location, glm::vec3(scale)) {}

Cube::Cube(const glm::vec3& location, const glm::vec3& scale)
{
    location_ = location;
    scale_ = scale;
}

void Cube::Draw(nextfloor::objects::RendererEngine* renderer)
{
    renderer->Draw(mvp_);
}

}  // namespace polygons

}  // namespace nextfloor
