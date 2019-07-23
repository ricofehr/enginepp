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

using nextfloor::objects::EngineGrid;
using nextfloor::objects::EngineGridBox;
using nextfloor::objects::EngineObject;

/**
 *  @class Grid
 *  @brief Abstract class who defines generic grid object
 */
class Grid : public EngineGrid {

public:

    /*
     *  Initial collision countdown value
     */
    static constexpr int kCOLLISION_COUNTDOWN = 4;

    Grid(EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension);

    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    virtual ~Grid() override
    {
        DeleteGrid();
    }

    virtual bool IsPositionEmpty(glm::ivec3 coords) const noexcept override;
    virtual bool IsFrontPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsRightPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsLeftPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsBackPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsFloorPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsRoofPositionFilled(glm::ivec3 coords) const noexcept override;
    virtual bool IsPositionFilled(glm::ivec3 coords) const noexcept override;

    virtual void DisplayGrid() const noexcept override;

    virtual void ComputePlacementsInGrid() noexcept override;

    virtual glm::vec3 CalculateAbsoluteCoordinates(glm::ivec3 coords) const noexcept override;

    virtual std::vector<EngineGridBox*> AddItemToGrid(EngineObject* object) noexcept override;

    virtual void RemoveItemToGrid(EngineObject* object) noexcept override;

    virtual void ResetGrid() noexcept override;

    virtual bool IsInside(glm::vec3 location_object) const noexcept override;

    virtual float width() const noexcept override final
    {
        return width_boxes_count() * box_width();
    }

    virtual float height() const noexcept override final
    {
        return height_boxes_count() * box_height();
    }

    virtual float depth() const noexcept override final
    {
        return depth_boxes_count() * box_depth();
    }

    virtual glm::vec3 scale() const noexcept override final
    {
        return glm::vec3(width()/2,
                         height()/2,
                         depth()/2);
    }

    virtual void lock() override final
    {
        grid_mutex_.lock();
    }

    virtual void unlock() override final
    {
        grid_mutex_.unlock();
    }

    virtual glm::vec3 CalculateFirstPointInGrid() const noexcept override final;

    virtual glm::ivec3 boxes_count() const override final
    {
        return glm::ivec3(width_boxes_count(), height_boxes_count(), depth_boxes_count());
    }

    virtual glm::vec3 box_dimension() const override final
    {
        return glm::vec3(box_width(), box_height(), box_depth());
    }

    virtual glm::vec3 CalculateFrontSideLocation() const noexcept override;
    virtual glm::vec3 CalculateFloorSideLocation() const noexcept override;
    virtual glm::vec3 CalculateRoofSideLocation() const noexcept override;
    virtual glm::vec3 CalculateRightSideLocation() const noexcept override;
    virtual glm::vec3 CalculateBackSideLocation() const noexcept override;
    virtual glm::vec3 CalculateLeftSideLocation() const noexcept override;
    virtual glm::vec3 CalculateFrontSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateRightSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateBackSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateLeftSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateFloorSideBorderScale() const noexcept override;
    virtual glm::vec3 CalculateRoofSideBorderScale() const noexcept override;

protected:

    void InitGrid() noexcept;
    void InitBoxes() noexcept;
    void DeleteGrid() noexcept;

    EngineGridBox* getGridBox(glm::ivec3 coords)
    {
        return boxes_[coords.x][coords.y][coords.z].get();
    }

    virtual std::unique_ptr<EngineGridBox> AllocateGridBox(glm::ivec3 coords) = 0;

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

    int type_{10000};

    glm::vec3 box_dimension_;

    glm::ivec3 boxes_count_;

private:

    EngineGridBox* AddItemToGrid(glm::ivec3 coords, EngineObject* object) noexcept;
    void RemoveItemToGrid(glm::ivec3 coords, EngineObject* object) noexcept;

    void ComputeFirstPointInGrid() noexcept;
    std::vector<EngineGridBox*> ParseGridForObjectPlacements(EngineObject *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept;
    glm::ivec3 PointToGridIndexes(glm::vec3 point) noexcept;
    glm::ivec3 CalculateLengthIndexes(glm::vec3 point_min, glm::vec3 point_max);
    bool IsCooordsAreCorrect(glm::ivec3 coords);

    tbb::mutex grid_mutex_;

    EngineObject* owner_;

    std::unique_ptr<EngineGridBox> ***boxes_;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_GRID_H_
