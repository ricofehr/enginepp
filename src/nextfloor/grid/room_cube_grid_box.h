/**
 *  @file room_cube_grid_box.h
 *  @brief RoomCubeGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMCUBEGRIDBOX_H_
#define NEXTFLOOR_GRID_ROOMCUBEGRIDBOX_H_

#include "nextfloor/grid/cube_grid_box.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomCubeGridBox
 *  @brief Abstract class who defines universe grid object
 */
class RoomCubeGridBox : public CubeGridBox {

public:

    RoomCubeGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner);
    RoomCubeGridBox(RoomCubeGridBox&&) = default;
    RoomCubeGridBox& operator=(RoomCubeGridBox&&) = default;
    RoomCubeGridBox(const RoomCubeGridBox&) = default;
    RoomCubeGridBox& operator=(const RoomCubeGridBox&) = default;
    virtual ~RoomCubeGridBox() = default;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_ROOMCUBEGRIDBOX_H_
