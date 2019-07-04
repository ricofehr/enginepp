/**
 *  @file brick.cc
 *  @brief Brick class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/brick.h"

#include <memory>

#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace objects {

Brick::Brick(float scale, glm::vec4 location)
{
    using nextfloor::polygons::Cube;

    type_ = kMODEL_BRICK;
    auto cube_ptr {std::make_unique<Cube>(scale, location)};
    meshes_.push_back(std::move(cube_ptr));
}

void Brick::InitRenderer(std::unique_ptr<EngineRenderer> renderer) noexcept
{
    renderer_ = std::move(renderer);
}

} // namespace objects

} // namespace nextfloor
