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
    auto factory = nextfloor::core::CommonServices::getFactory();
    grid_ = factory->MakeRoomGrid(this);
    border_ = factory->MakeBorder(location, grid_->scale());
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
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeRock(grid()->CalculateFirstPointInGrid() + relative_location));
}

void Room::AddFrontWall()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeFrontWall(grid()->CalculateFrontSideLocation(), grid()->CalculateFrontSideBorderScale()));
}

void Room::AddRightWall()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeRightWall(grid()->CalculateRightSideLocation(), grid()->CalculateRightSideBorderScale()));
}

void Room::AddBackWall()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeBackWall(grid()->CalculateBackSideLocation(), grid()->CalculateBackSideBorderScale()));
}

void Room::AddLeftWall()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeLeftWall(grid()->CalculateLeftSideLocation(), grid()->CalculateLeftSideBorderScale()));
}

void Room::AddFloor()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeFloor(grid()->CalculateBottomSideLocation(), grid()->CalculateBottomSideBorderScale()));
}

void Room::AddRoof()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    add_child(factory->MakeRoof(grid()->CalculateTopSideLocation(), grid()->CalculateTopSideBorderScale()));
}

}  // namespace objects

}  // namespace nextfloor
