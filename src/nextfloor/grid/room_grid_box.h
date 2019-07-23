/**
 *  @file room_grid_box.h
 *  @brief RoomGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMGRIDBOX_H_
#define NEXTFLOOR_GRID_ROOMGRIDBOX_H_

#include "nextfloor/grid/grid_box.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomGridBox
 *  @brief Abstract class who defines universe grid object
 */
class RoomGridBox : public GridBox {

public:

    RoomGridBox(glm::vec3 coords, nextfloor::objects::EngineGrid* owner);

    RoomGridBox(RoomGridBox&&) = default;

    RoomGridBox& operator=(RoomGridBox&&) = default;

    RoomGridBox(const RoomGridBox&) = default;

    RoomGridBox& operator=(const RoomGridBox&) = default;

    virtual ~RoomGridBox() = default;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_ROOMGRIDBOX_H_
