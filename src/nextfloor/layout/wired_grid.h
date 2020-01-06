/**
 *  @file wired_grid.h
 *  @brief WiredGrid class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_WIREDGRID_H_
#define NEXTFLOOR_GRID_WIREDGRID_H_

#include "nextfloor/playground/grid.h"

#include <glm/glm.hpp>
#include <memory>
#include <tbb/mutex.h>
#include <vector>

#include "nextfloor/mesh/mesh.h"
#include "nextfloor/mesh/grid_box.h"

namespace nextfloor {

namespace layout {

/**
 *  @class WiredGrid
 *  @brief Abstract class who defines generic grid object
 */
class WiredGrid : public nextfloor::playground::Grid {

public:
    ~WiredGrid() override = default;

    bool IsPositionEmpty(const glm::ivec3& coords) const final;
    bool IsFrontPositionFilled(const glm::ivec3& coords) const final;
    bool IsRightPositionFilled(const glm::ivec3& coords) const final;
    bool IsLeftPositionFilled(const glm::ivec3& coords) const final;
    bool IsBackPositionFilled(const glm::ivec3& coords) const final;
    bool IsBottomPositionFilled(const glm::ivec3& coords) const final;
    bool IsTopPositionFilled(const glm::ivec3& coords) const final;
    bool IsPositionFilled(const glm::ivec3& coords) const final;

    std::vector<nextfloor::mesh::Mesh*> FindCollisionNeighbors(const glm::vec3& coord) const final;

    glm::vec3 CalculateFirstPointInGrid() const final;
    glm::vec3 CalculateAbsoluteCoordinates(const glm::ivec3& coords) const final;

    void AddItem(nextfloor::mesh::Mesh* object) final;
    void RemoveMesh(nextfloor::mesh::Mesh* object) final;
    bool IsInside(const glm::vec3& location_object) const final;

    void DisplayGrid() const final;
    void ResetGrid() final;

    glm::vec3 CalculateFrontSideLocation() const final;
    glm::vec3 CalculateRightSideLocation() const final;
    glm::vec3 CalculateBackSideLocation() const final;
    glm::vec3 CalculateLeftSideLocation() const final;
    glm::vec3 CalculateBottomSideLocation() const final;
    glm::vec3 CalculateTopSideLocation() const final;

    glm::vec3 CalculateFrontSideBorderScale() const final;
    glm::vec3 CalculateRightSideBorderScale() const final;
    glm::vec3 CalculateBackSideBorderScale() const final;
    glm::vec3 CalculateLeftSideBorderScale() const final;
    glm::vec3 CalculateBottomSideBorderScale() const final;
    glm::vec3 CalculateTopSideBorderScale() const final;

    glm::vec3 scale() const final { return glm::vec3(width() / 2, height() / 2, depth() / 2); }

    glm::vec3 dimension() const final { return glm::vec3(width(), height(), depth()); }

protected:
    WiredGrid(const glm::vec3& location,
              const glm::ivec3& boxes_count,
              const glm::vec3& box_dimension,
              std::unique_ptr<nextfloor::mesh::GridBox>*** boxes);

    WiredGrid(WiredGrid&&) = delete;
    WiredGrid& operator=(WiredGrid&&) = delete;
    WiredGrid(const WiredGrid&) = delete;
    WiredGrid& operator=(const WiredGrid&) = delete;

    void ParseGridForObjectPlacements(nextfloor::mesh::Mesh* object);

    void DeleteGrid() noexcept;

    nextfloor::mesh::GridBox* getGridBox(const glm::ivec3& coords)
    {
        return boxes_[coords.x][coords.y][coords.z].get();
    }

    int width_boxes_count() const { return boxes_count_.x; }

    int height_boxes_count() const { return boxes_count_.y; }

    int depth_boxes_count() const { return boxes_count_.z; }

    float box_width() const { return box_dimension_.x; }

    float box_height() const { return box_dimension_.y; }

    float box_depth() const { return box_dimension_.z; }

    void lock() { mutex_.lock(); }

    void unlock() { mutex_.unlock(); }


private:
    static constexpr float kWallPadding = 0.25f;
    static constexpr float kWallDeepScale = 0.25f;

    void InitBoxes();

    std::vector<nextfloor::mesh::Mesh*> FindOccupants(const glm::ivec3& coords) const;
    nextfloor::mesh::GridBox* AddItemToGrid(const glm::ivec3& coords, nextfloor::mesh::Mesh* object);
    void RemoveItemToGrid(const glm::ivec3& coords, nextfloor::mesh::Mesh* object);

    glm::ivec3 PointToCoords(const glm::vec3& point) const;
    glm::ivec3 CalculateCoordsLengthBetweenPoints(const glm::vec3& point_min, const glm::vec3& point_max);
    bool IsCooordsAreCorrect(const glm::ivec3& coords) const;

    float width() const { return width_boxes_count() * box_width(); }

    float height() const { return height_boxes_count() * box_height(); }

    float depth() const { return depth_boxes_count() * box_depth(); }

    float min_box_side_dimension() const;

    std::vector<nextfloor::mesh::Mesh*> FindFrontPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontCenterPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontRightPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontRightBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontRightTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontLeftPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontLeftBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontLeftTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindFrontTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindRightPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindRightCenterPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindRightBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindRightTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackCenterPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackRightPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackRightBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackRightTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackLeftPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackLeftBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackLeftTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBackTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindLeftPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindLeftCenterPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindLeftBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindLeftTopPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindBottomPositionCollisionNeighbors(const glm::vec3& coords) const;
    std::vector<nextfloor::mesh::Mesh*> FindTopPositionCollisionNeighbors(const glm::vec3& coords) const;

    std::unique_ptr<nextfloor::mesh::GridBox>*** boxes_;
    glm::vec3 box_dimension_;
    glm::vec3 location_;
    glm::ivec3 boxes_count_;
    tbb::mutex mutex_;
};

}  // namespace layout

}  // namespace nextfloor

#endif  // NEXTFLOOR_GRID_WIREDGRID_H_
