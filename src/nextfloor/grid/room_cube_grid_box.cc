/**
 *  @file room_cube_grid_box.cc
 *  @brief RoomCubeGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_cube_grid_box.h"

namespace nextfloor {

namespace grid {


RoomCubeGridBox::RoomCubeGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner)
    : CubeGridBox(coords, owner) { }

} // namespace grid

} // namespace nextfloor
