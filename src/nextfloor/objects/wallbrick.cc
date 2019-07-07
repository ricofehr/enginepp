/**
 *  @file wall.cc
 *  @brief Wall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wallbrick.h"

#include <memory>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

WallBrick::WallBrick(glm::vec3 location, glm::vec3 scale, std::string texture)
{
    using nextfloor::core::CommonServices;

    type_ = kMODEL_WALL;
    auto cube = CommonServices::getFactory()->MakeCube(location, glm::vec3(scale));
    meshes_.push_back(std::move(cube));
    border_ = CommonServices::getFactory()->MakeBorder(location, glm::vec3(scale));
    renderer_ = CommonServices::getFactory()->MakeCubeRenderer(texture);
}

} // namespace objects

} // namespace nextfloor
