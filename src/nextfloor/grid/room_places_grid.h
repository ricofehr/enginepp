/**
 *  @file room_places_grid.h
 *  @brief RoomPlacesGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMPLACESGRID_H_
#define NEXTFLOOR_GRID_ROOMPLACESGRID_H_

#include "nextfloor/grid/places_grid.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomPlacesGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class RoomPlacesGrid : public PlacesGrid {

public:

    RoomPlacesGrid(nextfloor::objects::Mesh* owner);
    RoomPlacesGrid(RoomPlacesGrid&&) = default;
    RoomPlacesGrid& operator=(RoomPlacesGrid&&) = default;
    RoomPlacesGrid(const RoomPlacesGrid&) = delete;
    RoomPlacesGrid& operator=(const RoomPlacesGrid&) = delete;
    virtual ~RoomPlacesGrid() override;

protected:

    virtual std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

private:

    /*
     *  PlacesGrid Constants For Romm Setting
     */
    static constexpr int kWIDTH_BOXES_COUNT = 8;
    static constexpr int kHEIGHT_BOXES_COUNT = 6;
    static constexpr int kDEPTH_BOXES_COUNT = 8;
    static constexpr float kBOX_WIDTH = 2.0f;
    static constexpr float kBOX_HEIGHT = 2.0f;
    static constexpr float kBOX_DEPTH = 2.0f;

};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_ROOMPLACESGRID_H_
