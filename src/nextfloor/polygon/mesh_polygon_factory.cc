/**
 *  @file mesh_polygon_factory.cc
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/polygon/mesh_polygon_factory.h"

#include <glm/glm.hpp>

#include "nextfloor/polygon/cube.h"

namespace nextfloor {

namespace polygon {

std::unique_ptr<nextfloor::mesh::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                       float scale,
                                                                       const std::string& texture) const
{
    return MakeCube(location, glm::vec3(scale), texture);
}

std::unique_ptr<nextfloor::mesh::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                       const glm::vec3& scale) const
{
    return std::make_unique<Cube>(location, scale);
}

std::unique_ptr<nextfloor::mesh::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                       const glm::vec3& scale,
                                                                       const std::string& texture) const
{
    return std::make_unique<Cube>(location, scale, texture);
}


}  // namespace polygon

}  // namespace nextfloor