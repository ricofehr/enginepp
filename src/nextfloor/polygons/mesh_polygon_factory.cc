/**
 *  @file mesh_polygon_factory.cc
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/polygons/mesh_polygon_factory.h"

#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace polygons {

std::unique_ptr<nextfloor::objects::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location, float scale) const
{
    return MakeCube(location, glm::vec3(scale));
}

std::unique_ptr<nextfloor::objects::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                          float scale,
                                                                          const std::string& texture) const
{
    return MakeCube(location, glm::vec3(scale), texture);
}

std::unique_ptr<nextfloor::objects::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                          const glm::vec3& scale) const
{
    return std::make_unique<Cube>(location, scale);
}

std::unique_ptr<nextfloor::objects::Polygon> MeshPolygonFactory::MakeCube(const glm::vec3& location,
                                                                          const glm::vec3& scale,
                                                                          const std::string& texture) const
{
    return std::make_unique<Cube>(location, scale, texture);
}


}  // namespace polygons

}  // namespace nextfloor