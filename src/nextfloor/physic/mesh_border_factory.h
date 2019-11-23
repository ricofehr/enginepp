/**
 *  @file mesh_border_factory.h
 *  @brief MeshBorderFactory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PHYSIC_MESHBORDERFACTORY_H_
#define NEXTFLOOR_PHYSIC_MESHBORDERFACTORY_H_

#include "nextfloor/mesh/border_factory.h"

namespace nextfloor {

namespace physic {

/**
 *  @class MeshBorderFactory
 *  @brief Concrete factory for Border elements
 */
class MeshBorderFactory : public nextfloor::mesh::BorderFactory {

public:
    std::unique_ptr<nextfloor::mesh::Border> MakeBorder(const glm::vec3& location, float scale) const final;
    std::unique_ptr<nextfloor::mesh::Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const final;
};

}  // namespace physic

}  // namespace nextfloor

#endif  // NEXTFLOOR_PHYSIC_MESHBORDERFACTORY_H_
