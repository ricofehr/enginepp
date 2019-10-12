/**
 *  @file mesh_grid_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/grid/mesh_grid_factory.h"

#include "nextfloor/grid/room_grid.h"
#include "nextfloor/grid/universe_grid.h"
#include "nextfloor/grid/wired_grid_box.h"


namespace nextfloor {

namespace grid {

std::unique_ptr<nextfloor::objects::Grid> MeshGridFactory::MakeUniverseGrid(const glm::vec3& location) const
{
    return std::make_unique<UniverseGrid>(location,
                                          GenerateBoxes(UniverseGrid::kWIDTH_BOXES_COUNT,
                                                        UniverseGrid::kHEIGHT_BOXES_COUNT,
                                                        UniverseGrid::kDEPTH_BOXES_COUNT));
}

std::unique_ptr<nextfloor::objects::GridBox>*** MeshGridFactory::GenerateBoxes(unsigned int grid_width,
                                                                               unsigned int grid_height,
                                                                               unsigned int grid_depth) const
{
    std::unique_ptr<nextfloor::objects::GridBox>*** boxes
      = new std::unique_ptr<nextfloor::objects::GridBox>**[grid_width];
    assert(sizeof(boxes) == sizeof(void***));

    for (auto i = 0; i < grid_width; i++) {
        boxes[i] = new std::unique_ptr<nextfloor::objects::GridBox>*[grid_height];
        assert(sizeof(boxes[i]) == sizeof(void**));
        for (auto j = 0; j < grid_height; j++) {
            boxes[i][j] = new std::unique_ptr<nextfloor::objects::GridBox>[grid_depth];
            for (auto k = 0; k < grid_depth; k++) {
                boxes[i][j][k] = MakeGridBox(glm::ivec3(i, j, k));
            }
        }
    }

    return boxes;
}

std::unique_ptr<nextfloor::objects::Grid> MeshGridFactory::MakeRoomGrid(const glm::vec3& location) const
{
    return std::make_unique<RoomGrid>(
      location, GenerateBoxes(RoomGrid::kWIDTH_BOXES_COUNT, RoomGrid::kHEIGHT_BOXES_COUNT, RoomGrid::kDEPTH_BOXES_COUNT));
}

std::unique_ptr<nextfloor::objects::GridBox> MeshGridFactory::MakeGridBox(const glm::ivec3& coords) const
{
    return std::make_unique<WiredGridBox>(coords);
}

}  // namespace grid

}  // namespace nextfloor