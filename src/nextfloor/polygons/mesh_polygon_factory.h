/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MESHPOLYGONFACTORY_H_
#define NEXTFLOOR_OBJECTS_MESHPOLYGONFACTORY_H_

#include "nextfloor/objects/polygon_factory.h"

namespace nextfloor {

namespace polygons {

/**
 *  @class MeshFactory
 *  @brief Concret factory for Universe Objects
 */
class MeshPolygonFactory : public nextfloor::objects::PolygonFactory {

public:
    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location,
                                                          float scale,
                                                          const std::string& texture) const final;

    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location,
                                                          const glm::vec3& scale,
                                                          const std::string& texture) const final;
};

}  // namespace polygons

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MESHPOLYGONFACTORY_H_
