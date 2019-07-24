/**
 *  @file grid.h
 *  @brief Grid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_GRID_H_
#define NEXTFLOOR_GRID_GRID_H_

#include <memory>
#include <tbb/mutex.h>

#include "nextfloor/objects/engine_grid.h"

namespace nextfloor {

namespace grid {

/**
 *  @class Grid
 *  @brief Abstract class who defines generic grid object
 */
class Grid : public nextfloor::objects::EngineGrid {

public:

    Grid(nextfloor::objects::EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);
    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
    virtual ~Grid() = default;

    virtual bool IsPositionEmpty(glm::ivec3 coords) const noexcept override;
    virtual bool IsFrontPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsRightPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsLeftPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsBackPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsBottomPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsTopPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsPositionFilled(glm::ivec3 coords) const noexcept override;

    virtual glm::vec3 CalculateFirstPointInGrid() const noexcept override final;
    virtual void ComputePlacementsInGrid() noexcept override;
    virtual glm::vec3 CalculateAbsoluteCoordinates(glm::ivec3 coords) const noexcept override;

    virtual std::vector<nextfloor::objects::EngineGridBox*> AddItemToGrid(nextfloor::objects::EngineObject* object) noexcept override;
    virtual void RemoveItemToGrid(nextfloor::objects::EngineObject* object) noexcept override;
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

protected:

    virtual std::unique_ptr<nextfloor::objects::EngineGridBox> AllocateGridBox(glm::ivec3 coords) = 0;

    void InitBoxes() noexcept;
    void DeleteGrid() noexcept;

    nextfloor::objects::EngineGridBox* getGridBox(glm::ivec3 coords)
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

    nextfloor::objects::EngineGridBox* AddItemToGrid(glm::ivec3 coords, nextfloor::objects::EngineObject* object) noexcept;
    void RemoveItemToGrid(glm::ivec3 coords, nextfloor::objects::EngineObject* object) noexcept;
    std::vector<nextfloor::objects::EngineGridBox*> ParseGridForObjectPlacements(nextfloor::objects::EngineObject *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept;

    glm::ivec3 PointToCoords(glm::vec3 point) noexcept;
    glm::ivec3 CalculateCoordsLengthBetweenPoints(glm::vec3 point_min, glm::vec3 point_max);
    bool IsCooordsAreCorrect(glm::ivec3 coords);

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

    nextfloor::objects::EngineObject* owner_;
    std::unique_ptr<nextfloor::objects::EngineGridBox> ***boxes_;
    glm::vec3 box_dimension_;
    glm::ivec3 boxes_count_;
    tbb::mutex mutex_;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_GRID_H_
