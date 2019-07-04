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
     *  Grid Square Filled Constants
     */
    static constexpr int kGRID_UNKNOW = -1;
    static constexpr int kGRID_USED = 0;
    static constexpr int kGRID_EMPTY = 1;

    /*
     *  Grid Type Constants
     */
    static constexpr int kGRID_UNIVERSE = 0;
    static constexpr int kGRID_ROOM = 1;

    /*
     *  Initial collision countdown value
     */
    static constexpr int kCOLLISION_COUNTDOWN = 4;

    Grid(Grid&&) = default;
    Grid& operator=(Grid&&) = default;

    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;

    virtual ~Grid() override = default;

    virtual void InitDoorsAndWindows() noexcept override {}

    virtual int IsPositionInTheGridEmpty(glm::ivec3 box_coords_in_grid) const noexcept override;

    virtual void DisplayGrid() const noexcept override;

    //virtual std::vector<EngineObject*> FindItemsInGrid(glm::ivec3 box_coords_in_grid) const noexcept override;

    virtual void ComputePlacementsInGrid() noexcept override;

    virtual void AddItemToGrid(EngineObject* object) noexcept override;

    virtual void RemoveItemToGrid(glm::ivec3 box_coords_in_grid, EngineObject* object) noexcept override;

    virtual void ResetGrid() noexcept override;

    virtual bool IsInside(glm::vec3 location_object) const noexcept override;

    virtual float width_magnitude() const noexcept override final
    {
        return count_width_boxes() * x_length_by_box();
    }

    virtual float height_magnitude() const noexcept override final
    {
        return count_height_boxes() * y_length_by_box();
    }

    virtual float depth_magnitude() const noexcept override final
    {
        return count_depth_boxes() * z_length_by_box();
    }

    virtual glm::vec3 scale_vector() const noexcept override final
    {
        return glm::vec3(width_magnitude()/2,
                         height_magnitude()/2,
                         depth_magnitude()/2);
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

    virtual glm::ivec3 box_counts() const override final
    {
        return glm::ivec3(count_width_boxes(), count_height_boxes(), count_depth_boxes());
    }

    virtual glm::vec3 box_dimension() const override final
    {
        return glm::vec3(x_length_by_box(), y_length_by_box(), z_length_by_box());
    }

protected:

    Grid(EngineObject* owner);

    void InitGrid() noexcept;
    void InitBoxes() noexcept;
    void DeleteGrid() noexcept;
    void AddItemToGrid(glm::ivec3 box_coords_in_grid, EngineObject* object) noexcept;

    EngineGridBox* getGridBox(glm::ivec3 grid_coords)
    {
        return boxes_[grid_coords.x][grid_coords.y][grid_coords.z].get();
    }

    virtual std::unique_ptr<EngineGridBox> AllocateGridBox(glm::ivec3 grid_coords);

    virtual int count_width_boxes() const = 0;
    virtual int count_height_boxes() const = 0;
    virtual int count_depth_boxes() const = 0;

    virtual float x_length_by_box() const = 0;
    virtual float y_length_by_box() const = 0;
    virtual float z_length_by_box() const = 0;

    int type_{10000};

private:

    void ComputeFirstPointInGrid() noexcept;
    void ComputeGridUnits() noexcept;
    void ParseGridForObjectPlacements(EngineObject *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept;
    glm::ivec3 PointToGridIndexes(glm::vec3 point) noexcept;

    glm::vec3 grid_units_;

    tbb::mutex grid_mutex_;

    EngineObject* owner_;

    std::unique_ptr<EngineGridBox> ***boxes_;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_GRID_H_
