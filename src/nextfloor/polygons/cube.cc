/**
 *  @file cube.cc
 *  @brief Cube class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/cube.h"

#include "nextfloor/renderer/game_window.h"

#include <iostream>

namespace nextfloor {

namespace polygons {

Cube::Cube(float scale, glm::vec4 location)
    :Cube(glm::vec3(scale), location) {}

Cube::Cube(glm::vec3 scale, glm::vec4 location)
{
    location_ = location;
    scale_ = scale;
}

void Cube::Draw(nextfloor::objects::EngineRenderer* renderer) noexcept
{
    renderer->Draw(mvp_);
}

} // namespace graphics

} // namespace nextfloor
