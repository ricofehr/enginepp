/**
 *  @file universe_grid.h
 *  @brief UniverseGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_UNIVERSEGRID_H_
#define NEXTFLOOR_GRID_UNIVERSEGRID_H_

#include "nextfloor/grid/wired_grid.h"

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace grid {

/**
 *  @class UniverseGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class UniverseGrid : public WiredGrid {

public:
    UniverseGrid(nextfloor::objects::Mesh* owner, const nextfloor::objects::MeshFactory& factory);
    ~UniverseGrid() noexcept final;

protected:
    std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(const glm::ivec3& grid_coords,
                                                                 const nextfloor::objects::MeshFactory& factory) final;

private:
    /*
     *  WiredGrid Constants For Romm Setting
     */
    static constexpr int kWIDTH_BOXES_COUNT = 8;
    static constexpr int kHEIGHT_BOXES_COUNT = 4;
    static constexpr int kDEPTH_BOXES_COUNT = 8;
    static constexpr float kBOX_WIDTH = 16.0f;
    static constexpr float kBOX_HEIGHT = 12.0f;
    static constexpr float kBOX_DEPTH = 16.0f;
};

}  // namespace grid

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_UNIVERSEGRID_H_
