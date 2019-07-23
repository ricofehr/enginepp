/**
 *  @file object_grid.h
 *  @brief ObjectGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_OBJECTGRID_H_
#define NEXTFLOOR_GRID_OBJECTGRID_H_

#include "nextfloor/grid/grid.h"

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace grid {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class ObjectGrid : public Grid {

public:

    ObjectGrid(EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);

    ObjectGrid(ObjectGrid&&) = default;

    ObjectGrid& operator=(ObjectGrid&&) = default;

    ObjectGrid(const ObjectGrid&) = delete;

    ObjectGrid& operator=(const ObjectGrid&) = delete;

    virtual ~ObjectGrid() override;

protected:

    virtual std::unique_ptr<EngineGridBox> AllocateGridBox(glm::ivec3 grid_coords) override final;

};

} // namespace physics

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRID_H_
