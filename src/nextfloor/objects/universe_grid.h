/**
 *  @file universe_grid.h
 *  @brief UniverseGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_UNIVERSEGRID_H_
#define NEXTFLOOR_OBJECTS_UNIVERSEGRID_H_

#include "nextfloor/objects/grid.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace objects {

/**
 *  @class UniverseGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class UniverseGrid : public Grid {

public:

    UniverseGrid(EngineObject* owner);

    UniverseGrid(UniverseGrid&&) = default;

    UniverseGrid& operator=(UniverseGrid&&) = default;

    UniverseGrid(const UniverseGrid&) = delete;

    UniverseGrid& operator=(const UniverseGrid&) = delete;

    virtual ~UniverseGrid() override;

    void InitDoorsAndWindowsForRooms() noexcept;

protected:

    virtual std::unique_ptr<EngineGridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

    void InitLeftDoorAndWindowForRoom(glm::ivec3 coords) noexcept;
    void InitRightDoorAndWindowForRoom(glm::ivec3 coords) noexcept;
    void InitFrontDoorAndWindowForRoom(glm::ivec3 coords) noexcept;
    void InitBackDoorAndWindowForRoom(glm::ivec3 coords) noexcept;
    void InitRoofDoorAndWindowForRoom(glm::ivec3 coords) noexcept;
    void InitFloorDoorAndWindowForRoom(glm::ivec3 coords) noexcept;

    virtual int count_width_boxes() const override final
    {
        return kBOXES_IN_UNIVERSE_WIDTH;
    }

    virtual int count_height_boxes() const override final
    {
        return kBOXES_IN_UNIVERSE_HEIGHT;
    }

    virtual int count_depth_boxes() const override final
    {
        return kBOXES_IN_UNIVERSE_DEPTH;
    }

    virtual float x_length_by_box() const override final
    {
        return kX_LENGTH_BY_UNIVERSE_BOX;
    }

    virtual float y_length_by_box() const override final
    {
        return kY_LENGTH_BY_UNIVERSE_BOX;
    }

    virtual float z_length_by_box() const override final
    {
        return kZ_LENGTH_BY_UNIVERSE_BOX;
    }
private:

    /*
     *  Grid Constants For Romm Setting
     */
    static constexpr int kBOXES_IN_UNIVERSE_WIDTH = 8;
    static constexpr int kBOXES_IN_UNIVERSE_HEIGHT = 4;
    static constexpr int kBOXES_IN_UNIVERSE_DEPTH = 8;
    static constexpr float kX_LENGTH_BY_UNIVERSE_BOX = 16.0f;
    static constexpr float kY_LENGTH_BY_UNIVERSE_BOX = 12.0f;
    static constexpr float kZ_LENGTH_BY_UNIVERSE_BOX = 16.0f;
};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRID_H_
