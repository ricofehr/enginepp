/**
 *  @file universe_grid.h
 *  @brief UniverseGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_UNIVERSEGRID_H_
#define NEXTFLOOR_GRID_UNIVERSEGRID_H_

#include "nextfloor/layout/wired_grid.h"

#include <glm/glm.hpp>
#include <memory>

#include "nextfloor/mesh/grid_box.h"

namespace nextfloor {

namespace layout {

/**
 *  @class UniverseGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class UniverseGrid : public WiredGrid {

public:
    /*
     *  WiredGrid Constants For Romm Setting
     */
    static constexpr int kWidthBoxesCount = 8;
    static constexpr int kHeightBoxesCount = 4;
    static constexpr int kDepthBoxesCount = 8;
    static constexpr float kBoxWidth = 16.0f;
    static constexpr float kBoxHeight = 12.0f;
    static constexpr float kBoxDepth = 16.0f;

    UniverseGrid(const glm::vec3& location, std::unique_ptr<nextfloor::mesh::GridBox>*** boxes);
    ~UniverseGrid() noexcept final;
};

}  // namespace layout

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_UNIVERSEGRID_H_
