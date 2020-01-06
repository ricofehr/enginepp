/**
 *  @file mesh_grid_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/layout/mesh_grid_factory.h"

#include <memory>

#include "nextfloor/mesh/grid_box.h"

#include "nextfloor/layout/room_grid.h"
#include "nextfloor/layout/universe_grid.h"
#include "nextfloor/layout/wired_grid_box.h"

namespace nextfloor {

namespace layout {

std::unique_ptr<nextfloor::playground::Grid> MeshGridFactory::MakeUniverseGrid(const glm::vec3& location) const
{
    return std::make_unique<UniverseGrid>(
      location,
      GenerateBoxes(UniverseGrid::kWidthBoxesCount, UniverseGrid::kHeightBoxesCount, UniverseGrid::kDepthBoxesCount));
}

std::unique_ptr<nextfloor::playground::Grid> MeshGridFactory::MakeRoomGrid(const glm::vec3& location) const
{
    return std::make_unique<RoomGrid>(
      location, GenerateBoxes(RoomGrid::kWidthBoxesCount, RoomGrid::kHeightBoxesCount, RoomGrid::kDepthBoxesCount));
}


std::unique_ptr<nextfloor::mesh::GridBox>*** MeshGridFactory::GenerateBoxes(unsigned int grid_width,
                                                                            unsigned int grid_height,
                                                                            unsigned int grid_depth) const
{
    std::unique_ptr<nextfloor::mesh::GridBox>*** boxes = new std::unique_ptr<nextfloor::mesh::GridBox>**[grid_width];
    assert(sizeof(boxes) == sizeof(void***));

    for (auto i = 0; i < grid_width; i++) {
        boxes[i] = new std::unique_ptr<nextfloor::mesh::GridBox>*[grid_height];
        assert(sizeof(boxes[i]) == sizeof(void**));
        for (auto j = 0; j < grid_height; j++) {
            boxes[i][j] = new std::unique_ptr<nextfloor::mesh::GridBox>[grid_depth];
            for (auto k = 0; k < grid_depth; k++) {
                boxes[i][j][k] = MakeGridBox(glm::ivec3(i, j, k));
            }
        }
    }

    return boxes;
}

std::unique_ptr<nextfloor::mesh::GridBox> MeshGridFactory::MakeGridBox(const glm::ivec3& coords) const
{
    return std::make_unique<WiredGridBox>(coords);
}

}  // namespace layout

}  // namespace nextfloor