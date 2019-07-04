/**
 *  @file brick.cc
 *  @brief Brick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/brick.h"

#include <memory>

#include "nextfloor/objects/border.h"
#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace objects {

Brick::Brick(float scale, glm::vec4 location, std::unique_ptr<EngineRenderer> renderer)
{
    using nextfloor::polygons::Cube;

    type_ = kMODEL_BRICK;
    renderer_ = std::move(renderer);
    border_ = std::make_unique<Border>(scale, location); //, move);
    auto cube_ptr {std::make_unique<Cube>(scale, location)};
    meshes_.push_back(std::move(cube_ptr));
}

} // namespace objects

} // namespace nextfloor
