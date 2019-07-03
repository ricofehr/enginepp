/**
 *  @file grid.h
 *  @brief Grid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINEGRID_H_
#define NEXTFLOOR_OBJECTS_ENGINEGRID_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/engine_grid_box.h"
#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace objects {

/**
 *  @class EngineGrid
 *  @brief Abstract class who defines generic grid object
 */
class EngineGrid {

public:

    EngineGrid(EngineGrid&&) = default;

    EngineGrid& operator=(EngineGrid&&) = default;

    EngineGrid(const EngineGrid&) = delete;

    EngineGrid& operator=(const EngineGrid&) = delete;

    virtual ~EngineGrid() = default;

    virtual int IsPositionInTheGridEmpty(glm::ivec3 box_coords_in_grid) const noexcept = 0;

    virtual void DisplayGrid() const noexcept = 0;

    //virtual std::vector<EngineObject*> FindItemsInGrid(glm::ivec3 box_coords_in_grid) const noexcept = 0;

    virtual void ComputePlacementsInGrid() noexcept = 0;

    virtual void AddItemToGrid(EngineObject* object) noexcept = 0;

    virtual void RemoveItemToGrid(glm::ivec3 box_coords_in_grid, EngineObject* object) noexcept = 0;

    virtual void ResetGrid() noexcept = 0;

    virtual bool IsInside(glm::vec3 location_object) const noexcept = 0;

    virtual float width_magnitude() const noexcept = 0;

    virtual float height_magnitude() const noexcept = 0;

    virtual float depth_magnitude() const noexcept = 0;

    virtual glm::vec3 scale_vector() const noexcept = 0;

    virtual void lock() = 0;

    virtual void unlock() = 0;

    virtual glm::ivec3 box_counts() const = 0;
    virtual glm::vec3 box_dimension() const = 0;

    virtual glm::vec3 CalculateFirstPointInGrid() const noexcept = 0;

protected:

    EngineGrid() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINEGRID_H_
