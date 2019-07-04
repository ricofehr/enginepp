/**
 *  @file wall.cc
 *  @brief Wall class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/wall.h"

#include <memory>

#include "nextfloor/polygons/cube.h"

namespace nextfloor {

namespace objects {

Wall::Wall(glm::vec3 scale, glm::vec4 location)
{
    using nextfloor::polygons::Cube;

    type_ = kMODEL_WALL;
    auto cube_ptr {std::make_unique<Cube>(scale, location)};
    meshes_.push_back(std::move(cube_ptr));
}

void Wall::InitRenderer(std::unique_ptr<EngineRenderer> renderer) noexcept
{
    renderer_ = std::move(renderer);
}

} // namespace objects

} // namespace nextfloor
