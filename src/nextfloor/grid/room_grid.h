/**
 *  @file room_grid.h
 *  @brief RoomGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_ROOMGRID_H_
#define NEXTFLOOR_GRID_ROOMGRID_H_

#include "nextfloor/grid/grid.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class RoomGrid : public Grid {

public:

    RoomGrid(nextfloor::objects::EngineObject* owner);
    RoomGrid(RoomGrid&&) = default;
    RoomGrid& operator=(RoomGrid&&) = default;
    RoomGrid(const RoomGrid&) = delete;
    RoomGrid& operator=(const RoomGrid&) = delete;
    virtual ~RoomGrid() override;

protected:

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

private:

    /*
     *  Grid Constants For Romm Setting
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

#endif // NEXTFLOOR_OBJECTS_GRID_H_
