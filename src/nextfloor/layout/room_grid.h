/**
 *  @file room_grid.h
 *  @brief RoomGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMGRID_H_
#define NEXTFLOOR_GRID_ROOMGRID_H_

#include "nextfloor/layout/wired_grid.h"

#include <glm/glm.hpp>
#include <memory>

#include "nextfloor/mesh/grid_box.h"

namespace nextfloor {

namespace layout {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class RoomGrid : public WiredGrid {

public:
    /*
     *  WiredGrid Constants For Romm Setting
     */
    static constexpr int kWidthBoxesCount = 8;
    static constexpr int kHeightBoxesCount = 6;
    static constexpr int kDepthBoxesCount = 8;
    static constexpr float kBoxWidth = 2.0f;
    static constexpr float kBoxHeight = 2.0f;
    static constexpr float kBoxDepth = 2.0f;

    RoomGrid(const glm::vec3& location, std::unique_ptr<nextfloor::mesh::GridBox>*** boxes);
    ~RoomGrid() noexcept final;
};

}  // namespace layout

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_ROOMGRID_H_
