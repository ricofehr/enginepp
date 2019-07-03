/**
 *  @file room_grid.h
 *  @brief RoomGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ROOMGRID_H_
#define NEXTFLOOR_OBJECTS_ROOMGRID_H_

#include "nextfloor/objects/grid.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace objects {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class RoomGrid : public Grid {

public:

    RoomGrid(EngineObject* owner);

    RoomGrid(RoomGrid&&) = default;

    RoomGrid& operator=(RoomGrid&&) = default;

    RoomGrid(const RoomGrid&) = delete;

    RoomGrid& operator=(const RoomGrid&) = delete;

    ~RoomGrid();

protected:

    virtual std::unique_ptr<EngineGridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

    virtual int count_width_boxes() const override final
    {
        return kBOXES_IN_ROOM_WIDTH;
    }

    virtual int count_height_boxes() const override final
    {
        return kBOXES_IN_ROOM_HEIGHT;
    }

    virtual int count_depth_boxes() const override final
    {
        return kBOXES_IN_ROOM_DEPTH;
    }

    virtual float x_length_by_box() const override final
    {
        return kX_LENGTH_BY_ROOM_BOX;
    }

    virtual float y_length_by_box() const override final
    {
        return kY_LENGTH_BY_ROOM_BOX;
    }

    virtual float z_length_by_box() const override final
    {
        return kZ_LENGTH_BY_ROOM_BOX;
    }

private:

    /*
     *  Grid Constants For Romm Setting
     */
    static constexpr int kBOXES_IN_ROOM_WIDTH = 8;
    static constexpr int kBOXES_IN_ROOM_HEIGHT = 6;
    static constexpr int kBOXES_IN_ROOM_DEPTH = 8;
    static constexpr float kX_LENGTH_BY_ROOM_BOX = 2.0f;
    static constexpr float kY_LENGTH_BY_ROOM_BOX = 2.0f;
    static constexpr float kZ_LENGTH_BY_ROOM_BOX = 2.0f;

};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRID_H_
