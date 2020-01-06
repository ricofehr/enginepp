/**
 *  @file mesh_border_factory.cc
 *  @brief MeshBorderFactory Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/physic/mesh_border_factory.h"

#include <glm/glm.hpp>

#include "nextfloor/physic/cube_border.h"

namespace nextfloor {

namespace physic {

std::unique_ptr<nextfloor::mesh::Border> MeshBorderFactory::MakeBorder(const glm::vec3& location, float scale) const
{
    return MakeBorder(location, glm::vec3(scale));
}

std::unique_ptr<nextfloor::mesh::Border> MeshBorderFactory::MakeBorder(const glm::vec3& location,
                                                                       const glm::vec3& scale) const
{
    return std::make_unique<CubeBorder>(location, scale);
}

}  // namespace physic

}  // namespace nextfloor
