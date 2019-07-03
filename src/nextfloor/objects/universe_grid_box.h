/**
 *  @file grid_box.h
 *  @brief GridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_UNIVERSEGRIDBOX_H_
#define NEXTFLOOR_OBJECTS_UNIVERSEGRIDBOX_H_

#include "nextfloor/objects/grid_box.h"

namespace nextfloor {

namespace objects {

/**
 *  @class UniverseGridBox
 *  @brief Abstract class who defines universe grid object
 */
class UniverseGridBox : public GridBox {

public:

    UniverseGridBox(glm::vec3 coords, EngineGrid* owner);

    UniverseGridBox(UniverseGridBox&&) = default;

    UniverseGridBox& operator=(UniverseGridBox&&) = default;

    UniverseGridBox(const UniverseGridBox&) = default;

    UniverseGridBox& operator=(const UniverseGridBox&) = default;

    virtual ~UniverseGridBox() = default;

    void AddRoomDoor(int side);

    void AddRoomWindow(int side);
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_UNIVERSEGRIDBOX_H_
