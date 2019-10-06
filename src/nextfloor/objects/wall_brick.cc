/**
 *  @file wall_brick.cc
 *  @brief WallBrick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall_brick.h"

namespace nextfloor {

namespace objects {

WallBrick::WallBrick(const glm::vec3& location, const glm::vec3& scale, const std::string& texture, const MeshFactory& factory)
{
    polygons_.push_back(factory.MakeCube(location, scale, texture));
    border_ = factory.MakeBorder(location, scale);
}

}  // namespace objects

}  // namespace nextfloor
