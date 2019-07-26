/**
 *  @file room_grid_box.cc
 *  @brief RoomGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/room_grid_box.h"

namespace nextfloor {

namespace grid {


RoomGridBox::RoomGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner)
    : WiredGridBox(coords, owner) { }

} // namespace grid

} // namespace nextfloor
