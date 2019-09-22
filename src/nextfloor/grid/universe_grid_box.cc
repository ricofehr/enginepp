/**
 *  @file universe_grid_box.cc
 *  @brief UniverseGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/universe_grid_box.h"

namespace nextfloor {

namespace grid {


UniverseGridBox::UniverseGridBox(const glm::vec3& coords, nextfloor::objects::Grid* owner) : WiredGridBox(coords, owner)
{}

}  // namespace grid

}  // namespace nextfloor
