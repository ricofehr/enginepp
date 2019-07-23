/**
 *  @file universe_grid_box.cc
 *  @brief UniverseGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/universe_grid_box.h"

namespace nextfloor {

namespace grid {


UniverseGridBox::UniverseGridBox(glm::vec3 coords, nextfloor::objects::EngineGrid* owner)
    :GridBox(coords, owner) {}

} // namespace grid

} // namespace nextfloor
