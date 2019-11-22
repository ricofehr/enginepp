/**
 *  @file room.cc
 *  @brief Room class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/room.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace playground {

Room::Room(std::unique_ptr<Grid> grid,
           std::unique_ptr<nextfloor::mesh::Border> border,
           std::vector<std::unique_ptr<Wall>> walls,
           std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects)
{
    grid_ = std::move(grid);
    border_ = std::move(border);
    InitChilds(std::move(walls), std::move(objects));
    grid_->DisplayGrid();
}

void Room::InitChilds(std::vector<std::unique_ptr<Wall>> walls,
                      std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects)
{
    for (auto& wall : walls) {
        add_child(std::move(wall));
    }

    for (auto& object : objects) {
        add_child(std::move(object));
    }
}

// void Room::InitChildsIntoGrid()
// {
//     for (auto& object : objects_) {
//         if (object->hasNoChilds()) {
//             object->AddIntoAscendantGrid();
//         }
//         else {
//             object->InitChildsIntoGrid();
//         }
//     }
// }

}  // namespace playground

}  // namespace nextfloor
