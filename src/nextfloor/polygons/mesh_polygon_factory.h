/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_MESHPOLYGONFACTORY_H_
#define NEXTFLOOR_POLYGONS_MESHPOLYGONFACTORY_H_

#include "nextfloor/polygons/polygon_factory.h"

namespace nextfloor {

namespace polygons {

/**
 *  @class MeshFactory
 *  @brief Concret factory for Polygons
 */
class MeshPolygonFactory : public PolygonFactory {

public:
    std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, float scale, const std::string& texture) const final;

    std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Polygon> MakeCube(const glm::vec3& location,
                                      const glm::vec3& scale,
                                      const std::string& texture) const final;
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_MESHPOLYGONFACTORY_H_
