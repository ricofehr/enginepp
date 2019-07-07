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

    /*
     *  Mutators
     */
    void addDoor(int ind) { doors_[ind] = true; }
    void addWindow(int ind) { windows_[ind] = true; }

private:

    void AddFloor() noexcept;
    void AddLeftWall() noexcept;
    void AddRightWall() noexcept;
    void AddBackWall() noexcept;
    void AddFrontWall() noexcept;
    void AddRoof() noexcept;

    glm::vec3 CalculatePaddingFloor() const noexcept;
    glm::vec3 CalculatePaddingRoof() const noexcept;
    glm::vec3 CalculatePaddingLeftWall() const noexcept;
    glm::vec3 CalculatePaddingRightWall() const noexcept;
    glm::vec3 CalculatePaddingFrontWall() const noexcept;
    glm::vec3 CalculatePaddingBackWall() const noexcept;

    void AddFloorBrick(glm::vec3 location, glm::vec3 scale) noexcept;
    void AddRoofBrick(glm::vec3 location, glm::vec3 scale) noexcept;
    void AddSideWallBrick(glm::vec3 location, glm::vec3 scale) noexcept;

    glm::vec3 CalculateScaleWallYFixed() const noexcept;
    glm::vec3 CalculateScaleWallXFixed() const noexcept;
    glm::vec3 CalculateScaleWallZFixed() const noexcept;

    glm::vec3 CalculateBrickWallLocation(glm::ivec3 grid_coords, glm::vec3 padding) const noexcept;

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
