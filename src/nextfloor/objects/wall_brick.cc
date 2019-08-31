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

    polygons_.push_back(CommonServices::getMeshFactory()->MakeCube(location, scale));
    border_ = CommonServices::getMeshFactory()->MakeBorder(location, scale);
    renderer_ = CommonServices::getMeshFactory()->MakeCubeRenderer(texture);
}

}  // namespace objects

}  // namespace nextfloor
