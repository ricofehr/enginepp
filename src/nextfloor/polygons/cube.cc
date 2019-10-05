/**
 *  @file cube.cc
 *  @brief Cube class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace polygons {

Cube::Cube(const glm::vec3& location, float scale, const std::string& texture)
      : Cube(location, glm::vec3(scale), texture)
{}

Cube::Cube(const glm::vec3& location, const glm::vec3& scale) : Cube(location, glm::vec3(scale), kNOTEXTURE) {}

Cube::Cube(const glm::vec3& location, const glm::vec3& scale, const std::string& texture)
{
    location_ = location;
    scale_ = scale;
    texture_ = texture;
}

}  // namespace polygons

}  // namespace nextfloor
