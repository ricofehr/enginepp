/**
 *  @file room.h
 *  @brief Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOM_H_
#define NEXTFLOOR_OBJECTS_ROOM_H_

#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/objects/model_mesh.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Room
 *  @brief Define a Room, inherits Model abstract class
 */
class Room : public ModelMesh {

public:
    Room(glm::vec3 location);

    Room(Room&&) = default;
    Room& operator=(Room&&) = default;
    Room(const Room&) = delete;
    Room& operator=(const Room&) = delete;

    virtual ~Room() override = default;

    bool IsInsideWall(glm::ivec3 grid_coords) const noexcept;

    void AddRock(glm::vec3 relative_location) noexcept;

private:
    void AddWalls() noexcept;
    void AddFrontWall() noexcept;
    void AddRightWall() noexcept;
    void AddBackWall() noexcept;
    void AddLeftWall() noexcept;
    void AddFloor() noexcept;
    void AddRoof() noexcept;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ROOM_H_
