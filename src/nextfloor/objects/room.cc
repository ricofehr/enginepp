/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

namespace nextfloor {

namespace objects {

Room::Room(glm::vec4 location)
     :Room(location, std::vector<bool>(6, false), std::vector<bool>(6, false)){}

Room::Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows) {
    type_ = kMODEL_ROOM;
    doors_ = is_doors;
    windows_ = is_windows;
}

void Room::InitGrid(std::unique_ptr<EngineGrid> grid) noexcept
{
    grid_ = std::move(grid);
}

bool Room::IsInsideWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (grid_coords.x == 0) {
        return IsInsideLeftWall(grid_coords);
    }

    if (grid_coords.x == box_counts.x - 1) {
        return IsInsideRightWall(grid_coords);
    }

    if (grid_coords.y == 0) {
        return IsInsideFloor(grid_coords);
    }

    if (grid_coords.y == box_counts.y - 1) {
        return IsInsideRoof(grid_coords);
    }

    if (grid_coords.z == 0) {
        return IsInsideFrontWall(grid_coords);
    }

    if (grid_coords.z == box_counts.z - 1) {
        return IsInsideBackWall(grid_coords);
    }

    return false;
}

bool Room::IsInsideFloor(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (!doors_[kSIDE_FLOOR]) {
        return true;
    }

    if (grid_coords.x != 2*box_counts.x/4 && grid_coords.x != 3*box_counts.x/4) {
        return true;
    }

    if (grid_coords.z != 2*box_counts.z/4 && grid_coords.z != 3*box_counts.z/4) {
        return true;
    }

    return false;
}

bool Room::IsInsideRoof(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (!doors_[kSIDE_ROOF]) {
        return true;
    }

    if (grid_coords.x != 2*box_counts.x/4 && grid_coords.x != 3*box_counts.x/4) {
        return true;
    }

    if (grid_coords.z != 2*box_counts.z/4 && grid_coords.z != 3*box_counts.z/4) {
        return true;
    }

    return false;
}

bool Room::IsInsideRightWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (doors_[kSIDE_RIGHT]) {
        if (grid_coords.y <= box_counts.y/3 && grid_coords.z == 0) {
            return false;
        }
    }

    if (windows_[kSIDE_RIGHT]) {
        if (grid_coords.y == box_counts.y/3) {
            if (grid_coords.z == 2*box_counts.z/4 || grid_coords.z == 3*box_counts.z/4) {
                return false;
            }
        }
    }

    return true;
}

bool Room::IsInsideLeftWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (doors_[kSIDE_LEFT]) {
        if (grid_coords.y <= box_counts.y/3 && grid_coords.z == 0) {
            return false;
        }
    }

    if (windows_[kSIDE_LEFT]) {
        if (grid_coords.y == box_counts.y/3) {
            if (grid_coords.z == 2*box_counts.z/4 || grid_coords.z == 3*box_counts.z/4) {
                return false;
            }
        }
    }

    return true;
}

bool Room::IsInsideFrontWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (doors_[kSIDE_FRONT]) {
        if (grid_coords.y <= box_counts.y/3 && grid_coords.x == 0) {
            return false;
        }
    }

    if (windows_[kSIDE_FRONT]) {
        if (grid_coords.y == box_counts.y/3) {
            if (grid_coords.x == 2*box_counts.x/4 || grid_coords.x == 3*box_counts.x/4) {
                return false;
            }
        }
    }

    return true;
}

bool Room::IsInsideBackWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (doors_[kSIDE_BACK]) {
        if (grid_coords.y <= box_counts.y/3 && grid_coords.x == 0) {
            return false;
        }
    }

    if (windows_[kSIDE_BACK]) {
        if (grid_coords.y == box_counts.y/3) {
            if (grid_coords.x == 2*box_counts.x/4 || grid_coords.x == 3*box_counts.x/4) {
                return false;
            }
        }
    }

    return true;
}

} // namespace universe

} // namespace nextfloor
