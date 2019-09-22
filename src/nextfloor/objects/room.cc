/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Room::Room(const glm::vec3& location)
{
    using nextfloor::core::CommonServices;

    grid_ = CommonServices::getFactory().MakeRoomGrid(this);
    border_ = CommonServices::getFactory().MakeBorder(location, grid_->scale());
    AddWalls();
}

void Room::AddWalls()
{
    AddFrontWall();
    AddRightWall();
    AddBackWall();
    AddLeftWall();
    AddFloor();
    AddRoof();
}

void Room::InitChildsIntoGrid()
{
    for (auto& object : objects_) {
        if (object->hasNoChilds()) {
            object->AddIntoAscendantGrid();
        }
        else {
            object->InitChildsIntoGrid();
        }
    }
}

void Room::AddRock(const glm::vec3& relative_location)
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeRock(grid()->CalculateFirstPointInGrid() + relative_location));
}

void Room::AddFrontWall()
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeFrontWall(grid()->CalculateFrontSideLocation(),
                                                         grid()->CalculateFrontSideBorderScale()));
}

void Room::AddRightWall()
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeRightWall(grid()->CalculateRightSideLocation(),
                                                         grid()->CalculateRightSideBorderScale()));
}

void Room::AddBackWall()
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeBackWall(grid()->CalculateBackSideLocation(),
                                                        grid()->CalculateBackSideBorderScale()));
}

void Room::AddLeftWall()
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeLeftWall(grid()->CalculateLeftSideLocation(),
                                                        grid()->CalculateLeftSideBorderScale()));
}

void Room::AddFloor()
{
    using nextfloor::core::CommonServices;
    add_child(CommonServices::getFactory().MakeFloor(grid()->CalculateBottomSideLocation(),
                                                     grid()->CalculateBottomSideBorderScale()));
}

void Room::AddRoof()
{
    using nextfloor::core::CommonServices;
    add_child(
      CommonServices::getFactory().MakeRoof(grid()->CalculateTopSideLocation(), grid()->CalculateTopSideBorderScale()));
}

}  // namespace objects

}  // namespace nextfloor
