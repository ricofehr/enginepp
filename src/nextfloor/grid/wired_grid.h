/**
 *  @file wired_grid.h
 *  @brief WiredGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_WIREDGRID_H_
#define NEXTFLOOR_GRID_WIREDGRID_H_

#include <memory>
#include <tbb/mutex.h>

#include "nextfloor/objects/grid.h"

namespace nextfloor {

namespace grid {

/**
 *  @class WiredGrid
 *  @brief Abstract class who defines generic grid object
 */
class WiredGrid : public nextfloor::objects::Grid {

public:

    WiredGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);
    WiredGrid(WiredGrid&&) = default;
    WiredGrid& operator=(WiredGrid&&) = default;
    WiredGrid(const WiredGrid&) = delete;
    WiredGrid& operator=(const WiredGrid&) = delete;
    virtual ~WiredGrid() = default;

    virtual bool IsPositionEmpty(glm::ivec3 coords) const noexcept override;
    virtual bool IsFrontPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsRightPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsLeftPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsBackPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsBottomPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsTopPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsPositionFilled(glm::ivec3 coords) const noexcept override;

    virtual std::vector<nextfloor::objects::Mesh*> FindCollisionNeighbors(glm::vec3 coord) const noexcept override;

    virtual glm::vec3 CalculateFirstPointInGrid() const noexcept override final;
    virtual glm::vec3 CalculateAbsoluteCoordinates(glm::ivec3 coords) const noexcept override;

    virtual std::vector<nextfloor::objects::GridBox*> AddItemToGrid(nextfloor::objects::Mesh* object) noexcept override;
    virtual void RemoveItemToGrid(nextfloor::objects::Mesh* object) noexcept override;
    virtual bool IsInside(glm::vec3 location_object) const noexcept override;

    virtual void DisplayGrid() const noexcept override;
    virtual void ResetGrid() noexcept override;

    virtual glm::vec3 CalculateFrontSideLocation() const noexcept override;
    virtual glm::vec3 CalculateRightSideLocation() const noexcept override;
    virtual glm::vec3 CalculateBackSideLocation() const noexcept override;
    virtual glm::vec3 CalculateLeftSideLocation() const noexcept override;
    virtual glm::vec3 CalculateBottomSideLocation() const noexcept override;
    virtual glm::vec3 CalculateTopSideLocation() const noexcept override;

    virtual glm::vec3 CalculateFrontSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateRightSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateBackSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateLeftSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateBottomSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateTopSideBorderScale() const noexcept override;

    virtual glm::vec3 scale() const noexcept override final
    {
        return glm::vec3(width()/2,
                         height()/2,
                         depth()/2);
    }

    virtual glm::vec3 dimension() const noexcept override final
    {
        return glm::vec3(width(), height(), depth());
    }

protected:

    virtual std::unique_ptr<nextfloor::objects::GridBox> AllocateGridBox(glm::ivec3 coords) = 0;

    void InitBoxes() noexcept;
    void DeleteGrid() noexcept;

    nextfloor::objects::GridBox* getGridBox(glm::ivec3 coords)
    {
        return boxes_[coords.x][coords.y][coords.z].get();
    }

    virtual int width_boxes_count() const
    {
        return boxes_count_.x;
    }

    virtual int height_boxes_count() const
    {
        return boxes_count_.y;
    }

    virtual int depth_boxes_count() const
    {
        return boxes_count_.z;
    }

    virtual float box_width() const
    {
        return box_dimension_.x;
    }

    virtual float box_height() const
    {
        return box_dimension_.y;
    }

    virtual float box_depth() const
    {
        return box_dimension_.z;
    }

    void lock()
    {
        mutex_.lock();
    }

    void unlock()
    {
        mutex_.unlock();
    }


private:

    std::vector<nextfloor::objects::Mesh*> FindOccupants(glm::ivec3 coords) const noexcept;
    nextfloor::objects::GridBox* AddItemToGrid(glm::ivec3 coords, nextfloor::objects::Mesh* object) noexcept;
    void RemoveItemToGrid(glm::ivec3 coords, nextfloor::objects::Mesh* object) noexcept;
    std::vector<nextfloor::objects::GridBox*> ParseGridForObjectPlacements(nextfloor::objects::Mesh *object) noexcept;

    glm::ivec3 PointToCoords(glm::vec3 point) noexcept;
    glm::ivec3 CalculateCoordsLengthBetweenPoints(glm::vec3 point_min, glm::vec3 point_max);
    bool IsCooordsAreCorrect(glm::ivec3 coords) const;

    float width() const noexcept
    {
        return width_boxes_count() * box_width();
    }

    float height() const noexcept
    {
        return height_boxes_count() * box_height();
    }

    float depth() const noexcept
    {
        return depth_boxes_count() * box_depth();
    }

    float min_box_side_dimension() const noexcept;

    std::vector<nextfloor::objects::Mesh*> FindFrontPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindFrontTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindRightCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBackTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindLeftCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept;
    std::vector<nextfloor::objects::Mesh*> FindTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept;

    nextfloor::objects::Mesh* owner_;
    std::unique_ptr<nextfloor::objects::GridBox> ***boxes_;
    glm::vec3 box_dimension_;
    glm::ivec3 boxes_count_;
    tbb::mutex mutex_;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_WIREDGRID_H_
