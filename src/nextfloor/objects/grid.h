/**
 *  @file grid.h
 *  @brief Grid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_GRID_H_
#define NEXTFLOOR_OBJECTS_GRID_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/grid_box.h"
#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Grid
 *  @brief Abstract class who defines generic grid object
 */
class Grid {

public:

    virtual ~Grid() = default;

    virtual bool IsPositionEmpty(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsFrontPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsRightPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsLeftPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsBackPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsBottomPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsTopPositionFilled(glm::ivec3 coords) const noexcept = 0;
    virtual bool IsPositionFilled(glm::ivec3 coords) const noexcept = 0;

    virtual glm::vec3 CalculateFirstPointInGrid() const noexcept = 0;
    virtual void ComputePlacementsInGrid() noexcept = 0;
    virtual glm::vec3 CalculateAbsoluteCoordinates(glm::ivec3 coords) const = 0;

    virtual std::vector<GridBox*> AddItemToGrid(Mesh* object) noexcept = 0;
    virtual void RemoveItemToGrid(Mesh* object) noexcept = 0;
    virtual bool IsInside(glm::vec3 location_object) const noexcept = 0;

    virtual void DisplayGrid() const noexcept = 0;
    virtual void ResetGrid() noexcept = 0;

    virtual glm::vec3 CalculateFrontSideLocation() const noexcept = 0;
    virtual glm::vec3 CalculateRightSideLocation() const noexcept = 0;
    virtual glm::vec3 CalculateBackSideLocation() const noexcept = 0;
    virtual glm::vec3 CalculateLeftSideLocation() const noexcept = 0;
    virtual glm::vec3 CalculateBottomSideLocation() const noexcept = 0;
    virtual glm::vec3 CalculateTopSideLocation() const noexcept = 0;

    virtual glm::vec3 CalculateFrontSideBorderScale() const noexcept = 0;
    virtual glm::vec3 CalculateRightSideBorderScale() const noexcept = 0;
    virtual glm::vec3 CalculateBackSideBorderScale() const noexcept = 0;
    virtual glm::vec3 CalculateLeftSideBorderScale() const noexcept = 0;
    virtual glm::vec3 CalculateBottomSideBorderScale() const noexcept = 0;
    virtual glm::vec3 CalculateTopSideBorderScale() const noexcept = 0;

    virtual glm::vec3 scale() const noexcept = 0;
    virtual glm::vec3 dimension() const noexcept = 0;

protected:

    Grid() = default;

    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRID_H_
