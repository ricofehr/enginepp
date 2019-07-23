/**
 *  @file wall_brick.cc
 *  @brief WallBrick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall_brick.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

WallBrick::WallBrick(glm::vec3 location, glm::vec3 scale, std::string texture)
{
    using nextfloor::core::CommonServices;

    meshes_.push_back(CommonServices::getFactory()->MakeCube(location, scale));
    border_ = CommonServices::getFactory()->MakeBorder(location, scale);
    renderer_ = CommonServices::getFactory()->MakeCubeRenderer(texture);
}

} // namespace objects

} // namespace nextfloor
