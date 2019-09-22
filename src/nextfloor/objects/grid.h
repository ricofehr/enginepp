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

    virtual bool IsPositionEmpty(const glm::ivec3& coords) const = 0;
    virtual bool IsFrontPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsRightPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsLeftPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsBackPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsBottomPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsTopPositionFilled(const glm::ivec3& coords) const = 0;
    virtual bool IsPositionFilled(const glm::ivec3& coords) const = 0;

    virtual std::vector<nextfloor::objects::Mesh*> FindCollisionNeighbors(const glm::vec3& coord) const = 0;
    virtual glm::vec3 CalculateFirstPointInGrid() const = 0;
    virtual glm::vec3 CalculateAbsoluteCoordinates(const glm::ivec3& coords) const = 0;

    virtual std::vector<GridBox*> AddItem(Mesh* object) = 0;
    virtual void RemoveMesh(Mesh* object) = 0;
    virtual bool IsInside(const glm::vec3& location_object) const = 0;

    virtual void DisplayGrid() const = 0;
    virtual void ResetGrid() = 0;

    virtual glm::vec3 CalculateFrontSideLocation() const = 0;
    virtual glm::vec3 CalculateRightSideLocation() const = 0;
    virtual glm::vec3 CalculateBackSideLocation() const = 0;
    virtual glm::vec3 CalculateLeftSideLocation() const = 0;
    virtual glm::vec3 CalculateBottomSideLocation() const = 0;
    virtual glm::vec3 CalculateTopSideLocation() const = 0;

    virtual glm::vec3 CalculateFrontSideBorderScale() const = 0;
    virtual glm::vec3 CalculateRightSideBorderScale() const = 0;
    virtual glm::vec3 CalculateBackSideBorderScale() const = 0;
    virtual glm::vec3 CalculateLeftSideBorderScale() const = 0;
    virtual glm::vec3 CalculateBottomSideBorderScale() const = 0;
    virtual glm::vec3 CalculateTopSideBorderScale() const = 0;

    virtual glm::vec3 scale() const = 0;
    virtual glm::vec3 dimension() const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_GRID_H_
