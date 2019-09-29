/**
 *  @file wall_brick.cc
 *  @brief WallBrick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall_brick.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

WallBrick::WallBrick(const glm::vec3& location, const glm::vec3& scale, const std::string& texture)
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    polygons_.push_back(factory->MakeCube(location, scale));
    border_ = factory->MakeBorder(location, scale);
    renderer_ = factory->MakeCubeRenderer(texture);
}

}  // namespace objects

}  // namespace nextfloor
