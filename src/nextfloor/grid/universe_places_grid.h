/**
 *  @file universe_places_grid.h
 *  @brief UniversePlacesGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_UNIVERSEPLACESGRID_H_
#define NEXTFLOOR_GRID_UNIVERSEPLACESGRID_H_

#include "nextfloor/grid/places_grid.h"

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class UniversePlacesGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class UniversePlacesGrid : public PlacesGrid {

public:

    UniversePlacesGrid(nextfloor::objects::Mesh* owner);
    UniversePlacesGrid(UniversePlacesGrid&&) = default;
    UniversePlacesGrid& operator=(UniversePlacesGrid&&) = default;
    UniversePlacesGrid(const UniversePlacesGrid&) = delete;
    UniversePlacesGrid& operator=(const UniversePlacesGrid&) = delete;
    virtual ~UniversePlacesGrid() override;

protected:

    virtual std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

private:

    /*
     *  PlacesGrid Constants For Romm Setting
     */
    static constexpr int kWIDTH_BOXES_COUNT = 8;
    static constexpr int kHEIGHT_BOXES_COUNT = 4;
    static constexpr int kDEPTH_BOXES_COUNT = 8;
    static constexpr float kBOX_WIDTH = 16.0f;
    static constexpr float kBOX_HEIGHT = 12.0f;
    static constexpr float kBOX_DEPTH = 16.0f;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_UNIVERSEPLACESGRID_H_
