/**
 *  @file room_grid_box.h
 *  @brief RoomGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMGRIDBOX_H_
#define NEXTFLOOR_GRID_ROOMGRIDBOX_H_

#include "nextfloor/grid/wired_grid_box.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomGridBox
 *  @brief Abstract class who defines universe grid object
 */
class RoomGridBox : public WiredGridBox {

public:
    RoomGridBox(const glm::vec3& coords, nextfloor::objects::Grid* owner);
    ~RoomGridBox() final = default;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_ROOMGRIDBOX_H_
