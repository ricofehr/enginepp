/**
 *  @file room_grid.h
 *  @brief RoomGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMGRID_H_
#define NEXTFLOOR_GRID_ROOMGRID_H_

#include "nextfloor/grid/wired_grid.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class RoomGrid : public WiredGrid {

public:
    RoomGrid(nextfloor::objects::Mesh* owner, const nextfloor::objects::MeshFactory& factory);
    ~RoomGrid() noexcept final;

protected:
    std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(const glm::ivec3& grid_coords,
                                                                 const nextfloor::objects::MeshFactory& factory) final;

private:
    /*
     *  WiredGrid Constants For Romm Setting
     */
    static constexpr int kWIDTH_BOXES_COUNT = 8;
    static constexpr int kHEIGHT_BOXES_COUNT = 6;
    static constexpr int kDEPTH_BOXES_COUNT = 8;
    static constexpr float kBOX_WIDTH = 2.0f;
    static constexpr float kBOX_HEIGHT = 2.0f;
    static constexpr float kBOX_DEPTH = 2.0f;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_ROOMGRID_H_
