/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

#include "nextfloor/objects/wall.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Room::Room(glm::vec3 location) {
    using nextfloor::core::CommonServices;

    grid_ = CommonServices::getFactory()->MakeRoomGrid(this);
    border_ = CommonServices::getFactory()->MakeBorder(location, grid_->scale());

    AddWalls();
    AddCamera();
    AddRock(glm::vec3(3.0f));

    grid_->DisplayGrid();
}

void Room::AddWalls() noexcept
{
    AddFrontWall();
    AddRightWall();
    AddBackWall();
    AddLeftWall();
    AddFloor();
    AddRoof();
}

void Room::AddCamera() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeCamera(glm::vec3(location().x, location().y-2.0f, location().z + 7.0f)));
}

void Room::AddRock(glm::vec3 relative_location) noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeRock(grid_->CalculateFirstPointInGrid() + relative_location));
}

void Room::AddFrontWall() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeFrontWall(grid()->CalculateFrontSideLocation(), grid()->CalculateFrontSideBorderScale()));
}


void Room::AddRightWall() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeRightWall(grid()->CalculateRightSideLocation(), grid()->CalculateRightSideBorderScale()));
}

void Room::AddBackWall() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeBackWall(grid()->CalculateBackSideLocation(), grid()->CalculateBackSideBorderScale()));
}

void Room::AddLeftWall() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeLeftWall(grid()->CalculateLeftSideLocation(), grid()->CalculateLeftSideBorderScale()));
}

void Room::AddFloor() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeFloor(grid()->CalculateFloorSideLocation(), grid()->CalculateFloorSideBorderScale()));
}

void Room::AddRoof() noexcept
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory()->MakeRoof(grid()->CalculateRoofSideLocation(), grid()->CalculateRoofSideBorderScale()));
}

} // namespace objects

} // namespace nextfloor
