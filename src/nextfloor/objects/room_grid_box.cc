/**
 *  @file room_grid_box.cc
 *  @brief RoomGridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room_grid_box.h"

namespace nextfloor {

namespace objects {


RoomGridBox::RoomGridBox(glm::vec3 coords, EngineGrid* owner)
:GridBox(coords, owner) { }

} // namespace objects

} // namespace nextfloor
