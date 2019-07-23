/**
 *  @file room.h
 *  @brief Room class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOM_H_
#define NEXTFLOOR_OBJECTS_ROOM_H_

#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/objects/model.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Room
 *  @brief Define a Room (inside with walls and moving objects), inherits Model3D abstract class
 */
class Room : public Model {

public:

    /*
     *  Wall Side constants
     */
    static constexpr int kSIDE_FRONT = 0;
    static constexpr int kSIDE_RIGHT = 1;
    static constexpr int kSIDE_BACK = 2;
    static constexpr int kSIDE_LEFT = 3;
    static constexpr int kSIDE_FLOOR = 4;
    static constexpr int kSIDE_ROOF = 5;

    Room(glm::vec3 location);

    Room(Room&&) = default;
    Room& operator=(Room&&) = default;

    Room(const Room&) = delete;
    Room& operator=(const Room&) = delete;

    ~Room() override = default;

    bool IsInsideWall(glm::ivec3 grid_coords) const noexcept;

    void AddCamera() noexcept;
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

} // namespace objects

} // namespace nextfloor

#endif //NEXTFLOOR_OBJECTS_ROOM_H_
