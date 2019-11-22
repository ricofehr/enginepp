/**
 *  @file polygon_factory.h
 *  @brief Abstract Factory Class for polygons
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGON_POLYGONFACTORY_H_
#define NEXTFLOOR_POLYGON_POLYGONFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/mesh/polygon.h"

namespace nextfloor {

namespace polygon {

/**
 *  @class MeshFactory
 *  @brief Abstract Factory Pattern for Polygons
 */
class PolygonFactory {

public:
    virtual ~PolygonFactory() = default;

    virtual std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location,
                                                               float scale,
                                                               const std::string& texture) const = 0;

    virtual std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location,
                                                               const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<nextfloor::mesh::Polygon> MakeCube(const glm::vec3& location,
                                                               const glm::vec3& scale,
                                                               const std::string& texture) const = 0;
};

}  // namespace polygon

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGON_POLYGONFACTORY_H_
