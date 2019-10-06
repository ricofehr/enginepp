/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Room::Room(const glm::vec3& location, const MeshFactory& factory)
{
    grid_ = factory.MakeRoomGrid(this);
    border_ = factory.MakeBorder(location, grid_->scale());
    AddWalls(factory);
}

void Room::AddWalls(const MeshFactory& factory)
{
    AddFrontWall(factory);
    AddRightWall(factory);
    AddBackWall(factory);
    AddLeftWall(factory);
    AddFloor(factory);
    AddRoof(factory);
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

void Room::AddRock(const glm::vec3& relative_location, const MeshFactory& factory)
{
    add_child(factory.MakeRock(grid()->CalculateFirstPointInGrid() + relative_location));
}

void Room::AddFrontWall(const MeshFactory& factory)
{
    add_child(factory.MakeFrontWall(grid()->CalculateFrontSideLocation(), grid()->CalculateFrontSideBorderScale()));
}

void Room::AddRightWall(const MeshFactory& factory)
{
    add_child(factory.MakeRightWall(grid()->CalculateRightSideLocation(), grid()->CalculateRightSideBorderScale()));
}

void Room::AddBackWall(const MeshFactory& factory)
{
    add_child(factory.MakeBackWall(grid()->CalculateBackSideLocation(), grid()->CalculateBackSideBorderScale()));
}

void Room::AddLeftWall(const MeshFactory& factory)
{
    add_child(factory.MakeLeftWall(grid()->CalculateLeftSideLocation(), grid()->CalculateLeftSideBorderScale()));
}

void Room::AddFloor(const MeshFactory& factory)
{
    add_child(factory.MakeFloor(grid()->CalculateBottomSideLocation(), grid()->CalculateBottomSideBorderScale()));
}

void Room::AddRoof(const MeshFactory& factory)
{
    add_child(factory.MakeRoof(grid()->CalculateTopSideLocation(), grid()->CalculateTopSideBorderScale()));
}

}  // namespace objects

}  // namespace nextfloor
