/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGON_MESHPOLYGONFACTORY_H_
#define NEXTFLOOR_POLYGON_MESHPOLYGONFACTORY_H_

#include "nextfloor/mesh/polygon_factory.h"

namespace nextfloor {

namespace polygon {

/**
 *  @class MeshFactory
 *  @brief Concret factory for Polygons
 */
class MeshPolygonFactory : public nextfloor::mesh::PolygonFactory {

public:
    std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location,
                                                       float scale,
                                                       const std::string& texture) const final;

    std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location,
                                                       const glm::vec3& scale,
                                                       const std::string& texture) const final;
};

}  // namespace polygon

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGON_MESHPOLYGONFACTORY_H_
