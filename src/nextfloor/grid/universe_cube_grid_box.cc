/**
 *  @file universe_grid_box.cc
 *  @brief UniverseGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/universe_cube_grid_box.h"

namespace nextfloor {

namespace grid {


UniverseCubeGridBox::UniverseCubeGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner)
    : CubeGridBox(coords, owner) {}

} // namespace grid

} // namespace nextfloor
