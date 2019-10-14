/**
 *  @file polygon_factory.h
 *  @brief Abstract Factory Class for polygons
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_POLYGONS_POLYGONFACTORY_H_
#define NEXTFLOOR_POLYGONS_POLYGONFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/polygons/polygon.h"

namespace nextfloor {

namespace polygons {

/**
 *  @class MeshFactory
 *  @brief Abstract Factory Pattern for Polygons
 */
class PolygonFactory {

public:
    virtual ~PolygonFactory() = default;

    virtual std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, float scale, const std::string& texture) const = 0;

    virtual std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Polygon> MakeCube(const glm::vec3& location,
                                              const glm::vec3& scale,
                                              const std::string& texture) const = 0;
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_POLYGONS_POLYGONFACTORY_H_
