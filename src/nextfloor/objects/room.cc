/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Room::Room(glm::vec3 location) {
    using nextfloor::core::CommonServices;

    type_ = kMODEL_ROOM;
    grid_ = CommonServices::getFactory()->MakeRoomGrid(this);
    border_ = CommonServices::getFactory()->MakeBorder(location, grid_->scale_vector());

    add_child(CommonServices::getFactory()->MakeCamera(glm::vec3(location.x, location.y, location.z + 5.0f)));

    AddFloor();
    AddLeftWall();
    AddRightWall();
    AddBackWall();
    AddFrontWall();
    AddRoof();

    add_child(CommonServices::getFactory()->MakeRock(grid_->CalculateFirstPointInGrid() + glm::vec3(3.0f)));
    grid_->DisplayGrid();
}

void Room::AddFloor() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto grid_coords = glm::ivec3(j,0,k);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingFloor());
                AddFloorBrick(location, CalculateScaleWallYFixed());
            }
        });
    });
}

glm::vec3 Room::CalculatePaddingFloor() const noexcept
{
    return CalculateScaleWallYFixed();
}

void Room::AddFloorBrick(glm::vec3 location, glm::vec3 scale) noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeFloorBrick(location, scale));
}

void Room::AddRoof() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto grid_coords = glm::ivec3(j,box_counts.y,k);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingRoof());
                AddRoofBrick(location, CalculateScaleWallYFixed());
            }
        });
    });
}

void Room::AddRoofBrick(glm::vec3 location, glm::vec3 scale) noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeRoofBrick(location, scale));
}

glm::vec3 Room::CalculatePaddingRoof() const noexcept
{
    auto box_dimension = grid_->box_dimension();
    return CalculateScaleWallYFixed() - glm::vec3(0.0f, box_dimension.y/2, 0.0f);
}

glm::vec3 Room::CalculateScaleWallYFixed() const noexcept
{
    auto box_counts = grid_->box_counts();
    auto box_dimension = grid_->box_dimension();
    return glm::vec3((box_counts.x/8)*box_dimension.x, box_dimension.y/4, (box_counts.z/8)*box_dimension.z);
}

void Room::AddLeftWall() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto grid_coords = glm::ivec3(0,j,k);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingLeftWall());
                AddSideWallBrick(location, CalculateScaleWallXFixed());
            }
        });
    });

}

glm::vec3 Room::CalculatePaddingLeftWall() const noexcept
{
    return CalculateScaleWallXFixed();
}


void Room::AddRightWall() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.z, box_counts.z/4, [&](int k) {
            auto grid_coords = glm::ivec3(box_counts.x,j,k);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingRightWall());
                AddSideWallBrick(location, CalculateScaleWallXFixed());
            }
        });
    });

}

glm::vec3 Room::CalculatePaddingRightWall() const noexcept
{
    auto box_dimension = grid_->box_dimension();
    return CalculateScaleWallXFixed() - glm::vec3(box_dimension.x/2, 0.0f, 0.0f);
}

glm::vec3 Room::CalculateScaleWallXFixed() const noexcept
{
    auto box_counts = grid_->box_counts();
    auto box_dimension = grid_->box_dimension();
    return glm::vec3(box_dimension.x/4, (box_counts.y/6)*box_dimension.y, (box_counts.z/8)*box_dimension.z);
}

void Room::AddFrontWall() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int k) {
            auto grid_coords = glm::ivec3(k,j,0);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingFrontWall());
                AddSideWallBrick(location, CalculateScaleWallZFixed());
            }
        });
    });
}

glm::vec3 Room::CalculatePaddingFrontWall() const noexcept
{
    return CalculateScaleWallZFixed();
}

void Room::AddBackWall() noexcept
{
    auto box_counts = grid_->box_counts();

    tbb::parallel_for (0, box_counts.y, box_counts.y/3, [&](int j) {
        tbb::parallel_for (0, box_counts.x, box_counts.x/4, [&](int k) {
            auto grid_coords = glm::ivec3(k,j,box_counts.z);
            if (IsInsideWall(grid_coords)) {
                auto location = CalculateBrickWallLocation(grid_coords, CalculatePaddingBackWall());
                AddSideWallBrick(location, CalculateScaleWallZFixed());
            }
        });
    });
}

glm::vec3 Room::CalculatePaddingBackWall() const noexcept
{
    auto box_dimension = grid_->box_dimension();
    return CalculateScaleWallZFixed() - glm::vec3(0.0f, 0.0f, box_dimension.z/2);
}

glm::vec3 Room::CalculateScaleWallZFixed() const noexcept
{
    auto box_counts = grid_->box_counts();
    auto box_dimension = grid_->box_dimension();
    return glm::vec3((box_counts.x/8)*box_dimension.x, (box_counts.y/6)*box_dimension.y, box_dimension.z/4);
}

glm::vec3 Room::CalculateBrickWallLocation(glm::ivec3 grid_coords, glm::vec3 padding) const noexcept
{
    return grid_->CalculateAbsoluteCoordinates(grid_coords) + padding;
}

void Room::AddSideWallBrick(glm::vec3 location, glm::vec3 scale) noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeSideWallBrick(location, scale));
}

bool Room::IsInsideWall(glm::ivec3 grid_coords) const noexcept
{
    auto box_counts = grid_->box_counts();

    if (grid_coords.x == 0) {
        return IsInsideLeftWall(grid_coords);
    }

    if (grid_coords.x == box_counts.x) {
        return IsInsideRightWall(grid_coords);
    }

    if (grid_coords.y == 0) {
        return IsInsideFloor(grid_coords);
    }

    if (grid_coords.y == box_counts.y) {
        return IsInsideRoof(grid_coords);
    }

    if (grid_coords.z == 0) {
        return IsInsideFrontWall(grid_coords);
    }

    if (grid_coords.z == box_counts.z) {
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

} // namespace objects

} // namespace nextfloor
