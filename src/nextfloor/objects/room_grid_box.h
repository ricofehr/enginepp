/**
 *  @file room_grid_box.h
 *  @brief RoomGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOMGRIDBOX_H_
#define NEXTFLOOR_OBJECTS_ROOMGRIDBOX_H_

#include "nextfloor/objects/grid_box.h"

namespace nextfloor {

namespace objects {

/**
 *  @class RoomGridBox
 *  @brief Abstract class who defines universe grid object
 */
class RoomGridBox : public GridBox {

public:

    RoomGridBox(glm::vec3 coords, EngineGrid* owner);

    RoomGridBox(RoomGridBox&&) = default;

    RoomGridBox& operator=(RoomGridBox&&) = default;

    RoomGridBox(const RoomGridBox&) = default;

    RoomGridBox& operator=(const RoomGridBox&) = default;

    virtual ~RoomGridBox() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ROOMGRIDBOX_H_
