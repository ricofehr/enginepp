/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Room::Room(std::unique_ptr<Grid> grid, std::unique_ptr<Border> border, std::vector<std::unique_ptr<Mesh>> childs)
{
    grid_ = std::move(grid);
    border_ = std::move(border);
    InitChilds(std::move(childs));
}

void Room::InitChilds(std::vector<std::unique_ptr<Mesh>> childs)
{
    for (auto& child : childs) {
        add_child(std::move(child));
    }
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

}  // namespace objects

}  // namespace nextfloor
