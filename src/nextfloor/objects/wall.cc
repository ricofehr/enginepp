/**
 *  @file wall.cc
 *  @brief Wall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall.h"

#include <memory>

#include "nextfloor/objects/border.h"
#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace objects {

Wall::Wall(glm::vec3 scale, glm::vec4 location, std::unique_ptr<EngineRenderer> renderer)
{
    using nextfloor::polygons::Cube;

    type_ = kMODEL_WALL;
    renderer_ = std::move(renderer);
    border_ = std::make_unique<Border>(scale, location);
    auto cube_ptr {std::make_unique<Cube>(scale, location)};
    meshes_.push_back(std::move(cube_ptr));
}

} // namespace objects

} // namespace nextfloor
