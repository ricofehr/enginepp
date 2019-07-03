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

    Room(glm::vec4 location);

    Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows);

    Room(Room&&) = default;

    Room& operator=(Room&&) = default;

    Room(const Room&) = delete;

    Room& operator=(const Room&) = delete;

    ~Room() override = default;

    bool IsInsideWall(glm::ivec3 grid_coords) const noexcept;

    /*
     *  Mutators
     */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

private:

    bool IsInsideFloor(glm::ivec3 grid_coords) const noexcept;
    bool IsInsideRoof(glm::ivec3 grid_coords) const noexcept;
    bool IsInsideRightWall(glm::ivec3 grid_coords) const noexcept;
    bool IsInsideLeftWall(glm::ivec3 grid_coords) const noexcept;
    bool IsInsideFrontWall(glm::ivec3 grid_coords) const noexcept;
    bool IsInsideBackWall(glm::ivec3 grid_coords) const noexcept;

    /** select room side with a Door */
    std::vector<bool> doors_{false, false, false, false, false, false};

    /** select room side with a Window */
    std::vector<bool> windows_{false, false, false, false, false, false};
};

} // namespace objects

} // namespace nextfloor

#endif //NEXTFLOOR_OBJECTS_ROOM_H_
