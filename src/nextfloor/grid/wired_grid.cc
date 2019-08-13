/**
 *  @file wired_grid.cc
 *  @brief Grid Implementation class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/wired_grid.h"

#include <tbb/tbb.h>
#include <iostream>

#include "nextfloor/grid/wired_grid_box.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace grid {

namespace {

glm::vec3 GetMinPointFromPoints(std::vector<glm::vec3> points)
{
    glm::vec3 point_min{100000.0f, 100000.0f, 100000.0f};

    for (auto &point : points) {
        if (point.x < point_min.x) {
            point_min.x = point.x;
        }

        if (point.y < point_min.y) {
            point_min.y = point.y;
        }

        if (point.z < point_min.z) {
            point_min.z = point.z;
        }
    }

    return point_min;
}

glm::vec3 GetMaxPointFromPoints(std::vector<glm::vec3> points)
{
    glm::vec3 point_max{-100000.0f, -100000.0f, -100000.0f};

    for (auto &point : points) {
        if (point.x > point_max.x) {
            point_max.x = point.x;
        }

        if (point.y > point_max.y) {
            point_max.y = point.y;
        }

        if (point.z > point_max.z) {
            point_max.z = point.z;
        }
    }

    return point_max;
}

}


WiredGrid::WiredGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension)
{
    owner_ = owner;
    boxes_count_ = boxes_count;
    box_dimension_ = box_dimension;
}

void WiredGrid::InitBoxes() noexcept
{
    using nextfloor::objects::GridBox;

    boxes_ = new std::unique_ptr<GridBox> **[width_boxes_count()];

    /* Ensure pointer is allocated */
    assert(sizeof(boxes_) == sizeof(void***));

    for (auto i = 0; i < width_boxes_count(); i++) {
        boxes_[i] = new std::unique_ptr<GridBox> *[height_boxes_count()];
        /* Ensure pointer is allocated */
        assert(sizeof(boxes_[i]) == sizeof(void**));

        for (auto j = 0; j < height_boxes_count(); j++) {
            boxes_[i][j] = new std::unique_ptr<GridBox>[depth_boxes_count()]();
            /* Ensure pointer is allocated */
            assert(sizeof(boxes_[i][j]) == sizeof(void*));

            /* Ensure each vector is allocated */
            for (auto k = 0; k < depth_boxes_count(); k++) {
                boxes_[i][j][k] = AllocateGridBox(glm::ivec3(i,j,k));
                assert(sizeof(boxes_[i][j][k]) == sizeof(std::unique_ptr<GridBox>));
            }
        }
    }
}

glm::vec3 WiredGrid::CalculateFirstPointInGrid() const noexcept
{
    return owner_->location() - glm::vec3(width()/2, height()/2, depth()/2);
}

glm::vec3 WiredGrid::CalculateAbsoluteCoordinates(glm::ivec3 coords) const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(coords.x * box_dimension_.x, coords.y * box_dimension_.y, coords.z * box_dimension_.z);
}

bool WiredGrid::IsPositionEmpty(glm::ivec3 coords) const noexcept
{
    return boxes_[coords.x][coords.y][coords.z]->IsEmpty();
}

bool WiredGrid::IsFrontPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.z != 0 && IsPositionFilled(glm::ivec3(coords.x,coords.y,coords.z-1));
}

bool WiredGrid::IsRightPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.x != (width_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x+1,coords.y,coords.z));
}

bool WiredGrid::IsLeftPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.x != 0 && IsPositionFilled(glm::ivec3(coords.x-1,coords.y,coords.z));
}

bool WiredGrid::IsBackPositionFilled(glm::ivec3 coords) const noexcept
{
    return (coords.z != depth_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x,coords.y,coords.z+1));
}

bool WiredGrid::IsBottomPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.y != 0 && IsPositionFilled(glm::ivec3(coords.x,coords.y-1,coords.z));
}

bool WiredGrid::IsTopPositionFilled(glm::ivec3 coords) const noexcept
{
    return (coords.y != height_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x,coords.y+1,coords.z));
}

bool WiredGrid::IsPositionFilled(glm::ivec3 coords) const noexcept
{
    return boxes_[coords.x][coords.y][coords.z]->IsFilled();
}


std::vector<nextfloor::objects::Mesh*> WiredGrid::FindCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors = FindOccupants(coords);

    auto front_neighbors = FindFrontPositionCollisionNeighbors(coords);
    auto right_neighbors = FindRightPositionCollisionNeighbors(coords);
    auto back_neighbors = FindBackPositionCollisionNeighbors(coords);
    auto left_neighbors = FindLeftPositionCollisionNeighbors(coords);
    auto bottom_neighbors = FindBottomPositionCollisionNeighbors(coords);
    auto top_neighbors = FindTopPositionCollisionNeighbors(coords);

    neighbors.insert(neighbors.end(), front_neighbors.begin(), front_neighbors.end());
    neighbors.insert(neighbors.end(), right_neighbors.begin(), right_neighbors.end());
    neighbors.insert(neighbors.end(), back_neighbors.begin(), back_neighbors.end());
    neighbors.insert(neighbors.end(), left_neighbors.begin(), left_neighbors.end());
    neighbors.insert(neighbors.end(), bottom_neighbors.begin(), bottom_neighbors.end());
    neighbors.insert(neighbors.end(), top_neighbors.begin(), top_neighbors.end());

    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);

    auto center_neighbors = FindFrontCenterPositionCollisionNeighbors(coords);
    auto right_neighbors = FindFrontRightPositionCollisionNeighbors(coords);
    auto right_bottom_neighbors = FindFrontRightBottomPositionCollisionNeighbors(coords);
    auto right_top_neighbors = FindFrontRightTopPositionCollisionNeighbors(coords);
    auto left_neighbors = FindFrontLeftPositionCollisionNeighbors(coords);
    auto left_bottom_neighbors = FindFrontLeftBottomPositionCollisionNeighbors(coords);
    auto left_top_neighbors = FindFrontLeftTopPositionCollisionNeighbors(coords);

    neighbors.insert(neighbors.end(), center_neighbors.begin(), center_neighbors.end());
    neighbors.insert(neighbors.end(), right_neighbors.begin(), right_neighbors.end());
    neighbors.insert(neighbors.end(), right_bottom_neighbors.begin(), right_bottom_neighbors.end());
    neighbors.insert(neighbors.end(), right_top_neighbors.begin(), right_top_neighbors.end());
    neighbors.insert(neighbors.end(), left_neighbors.begin(), left_neighbors.end());
    neighbors.insert(neighbors.end(), left_bottom_neighbors.begin(), left_bottom_neighbors.end());
    neighbors.insert(neighbors.end(), left_top_neighbors.begin(), left_top_neighbors.end());

    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y-1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y+1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y-1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y+1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y-1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindFrontTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y+1,coords.z-1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);

    auto center_neighbors = FindRightCenterPositionCollisionNeighbors(coords);
    auto bottom_neighbors = FindRightBottomPositionCollisionNeighbors(coords);
    auto top_neighbors = FindRightTopPositionCollisionNeighbors(coords);

    neighbors.insert(neighbors.end(), center_neighbors.begin(), center_neighbors.end());
    neighbors.insert(neighbors.end(), bottom_neighbors.begin(), bottom_neighbors.end());
    neighbors.insert(neighbors.end(), top_neighbors.begin(), top_neighbors.end());

    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindRightCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y-1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y+1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);

    auto center_neighbors = FindBackCenterPositionCollisionNeighbors(coords);
    auto right_neighbors = FindBackRightPositionCollisionNeighbors(coords);
    auto right_bottom_neighbors = FindBackRightBottomPositionCollisionNeighbors(coords);
    auto right_top_neighbors = FindBackRightTopPositionCollisionNeighbors(coords);
    auto left_neighbors = FindBackLeftPositionCollisionNeighbors(coords);
    auto left_bottom_neighbors = FindBackLeftBottomPositionCollisionNeighbors(coords);
    auto left_top_neighbors = FindBackLeftTopPositionCollisionNeighbors(coords);

    neighbors.insert(neighbors.end(), center_neighbors.begin(), center_neighbors.end());
    neighbors.insert(neighbors.end(), right_neighbors.begin(), right_neighbors.end());
    neighbors.insert(neighbors.end(), right_bottom_neighbors.begin(), right_bottom_neighbors.end());
    neighbors.insert(neighbors.end(), right_top_neighbors.begin(), right_top_neighbors.end());
    neighbors.insert(neighbors.end(), left_neighbors.begin(), left_neighbors.end());
    neighbors.insert(neighbors.end(), left_bottom_neighbors.begin(), left_bottom_neighbors.end());
    neighbors.insert(neighbors.end(), left_top_neighbors.begin(), left_top_neighbors.end());

    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackRightPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackRightBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y-1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackRightTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x-1,coords.y+1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y-1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y+1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y-1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBackTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y+1,coords.z+1);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindLeftPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);

    auto center_neighbors = FindLeftCenterPositionCollisionNeighbors(coords);
    auto bottom_neighbors = FindLeftBottomPositionCollisionNeighbors(coords);
    auto top_neighbors = FindLeftTopPositionCollisionNeighbors(coords);

    neighbors.insert(neighbors.end(), center_neighbors.begin(), center_neighbors.end());
    neighbors.insert(neighbors.end(), bottom_neighbors.begin(), bottom_neighbors.end());
    neighbors.insert(neighbors.end(), top_neighbors.begin(), top_neighbors.end());

    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindLeftCenterPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindLeftBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y-1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindLeftTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x+1,coords.y+1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindBottomPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y-1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindTopPositionCollisionNeighbors(glm::vec3 coords) const noexcept
{
    std::vector<nextfloor::objects::Mesh*> neighbors(0);
    auto neighbors_coord = glm::ivec3(coords.x,coords.y+1,coords.z);
    if (IsCooordsAreCorrect(neighbors_coord) && IsPositionFilled(neighbors_coord)) {
        neighbors = FindOccupants(neighbors_coord);
    }
    return neighbors;
}

std::vector<nextfloor::objects::Mesh*> WiredGrid::FindOccupants(glm::ivec3 coords) const noexcept
{
    return boxes_[coords.x][coords.y][coords.z]->occupants();
}

std::vector<nextfloor::objects::GridBox*> WiredGrid::AddItemToGrid(nextfloor::objects::Mesh* object) noexcept
{
    std::vector<glm::vec3> points = object->getCoordsModelMatrixComputed();

    auto point_min = GetMinPointFromPoints(points);
    auto point_max = GetMaxPointFromPoints(points);
    auto lengths = CalculateCoordsLengthBetweenPoints(point_min, point_max);

    return ParseGridForObjectPlacements(object, point_min, lengths);
}

glm::ivec3 WiredGrid::CalculateCoordsLengthBetweenPoints(glm::vec3 point_min, glm::vec3 point_max)
{
    int length_x = static_cast<int>(ceil((point_max.x - point_min.x) / box_dimension_.x));
    int length_y = static_cast<int>(ceil((point_max.y - point_min.y) / box_dimension_.y));
    int length_z = static_cast<int>(ceil((point_max.z - point_min.z) / box_dimension_.z));

    return glm::ivec3(length_x, length_y, length_z);
}

std::vector<nextfloor::objects::GridBox*> WiredGrid::ParseGridForObjectPlacements(nextfloor::objects::Mesh *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept
{
    std::vector<nextfloor::objects::GridBox*> coords_list;

    tbb::parallel_for (0, lengths.x, 1, [&](int cnt_x) {
        tbb::parallel_for (0, lengths.y, 1, [&](int cnt_y) {
            tbb::parallel_for (0, lengths.z, 1, [&](int cnt_z) {
                auto cnt = glm::vec3(cnt_x, cnt_y, cnt_z);
                auto intermediary_point = point_min + cnt * box_dimension_;
                auto coords = PointToCoords(intermediary_point);

                if (IsCooordsAreCorrect(coords)) {
                    coords_list.push_back(AddItemToGrid(coords, object));
                }
            });
        });
    });

    return coords_list;
}

glm::ivec3 WiredGrid::PointToCoords(glm::vec3 point) noexcept
{
    glm::vec3 grid0 = CalculateFirstPointInGrid();
    glm::vec3 coords = (glm::vec3(point.x, point.y, point.z) - grid0) / box_dimension_;

    return glm::ivec3(static_cast<int>(floor(coords.x)), static_cast<int>(floor(coords.y)), static_cast<int>(floor(coords.z)));
}

nextfloor::objects::GridBox* WiredGrid::AddItemToGrid(glm::ivec3 coords, nextfloor::objects::Mesh* object) noexcept
{
    assert(IsCooordsAreCorrect(coords));

    lock();
    boxes_[coords.x][coords.y][coords.z]->add(object);
    unlock();

    return boxes_[coords.x][coords.y][coords.z].get();
}

bool WiredGrid::IsCooordsAreCorrect(glm::ivec3 coords) const
{
    if (coords.x < 0 || coords.x >= width_boxes_count()) {
        return false;
    }

    if (coords.y < 0 || coords.y >= height_boxes_count()) {
        return false;
    }

    if (coords.z < 0 || coords.z >= depth_boxes_count()) {
        return false;
    }

    return true;
}

void WiredGrid::RemoveItemToGrid(nextfloor::objects::Mesh* object) noexcept
{
    for (auto i = 0; i < width_boxes_count(); i++) {
        for (auto j = 0; j < height_boxes_count(); j++) {
            for (auto k = 0; k < depth_boxes_count(); k++) {
                RemoveItemToGrid(glm::ivec3(i,j,k), object);
            }
        }
    }
}

void WiredGrid::RemoveItemToGrid(glm::ivec3 coords, nextfloor::objects::Mesh* object) noexcept
{
    boxes_[coords.x][coords.y][coords.z]->remove(object);
}

void WiredGrid::DisplayGrid() const noexcept
{
    std::string object_type{"MODEL3D"};

    std::cout << "=== GRID FOR " << object_type << " ID " << owner_->id() << " ===" << std::endl << std::endl;
    for (auto y = 0; y < height_boxes_count(); y++) {
        std::cout << "=== Floor " << y << std::endl;
        for (auto z = 0; z < depth_boxes_count(); z++) {
            for (auto x = 0; x < width_boxes_count(); x++) {
                std::cout << "  " << boxes_[x][y][z]->size();
            }

            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

void WiredGrid::ComputePlacementsInGrid() noexcept
{
    std::cout << "ComputePlacements" << std::endl;
}

glm::vec3 WiredGrid::CalculateFrontSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 WiredGrid::CalculateRightSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width() - 0.25f, height()/2, depth()/2);
}

glm::vec3 WiredGrid::CalculateBackSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height()/2, depth() - 0.25f);
}

glm::vec3 WiredGrid::CalculateLeftSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 WiredGrid::CalculateBottomSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, 0.25f, depth()/2);
}

glm::vec3 WiredGrid::CalculateTopSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height() - 0.25f, depth()/2);
}

glm::vec3 WiredGrid::CalculateFrontSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 WiredGrid::CalculateRightSideBorderScale() const noexcept
{
    return glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 WiredGrid::CalculateBackSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 WiredGrid::CalculateLeftSideBorderScale() const noexcept
{
    return glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 WiredGrid::CalculateBottomSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, 0.25f, depth()/2);
}

glm::vec3 WiredGrid::CalculateTopSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, 0.25f, depth()/2);
}

bool WiredGrid::IsInside(glm::vec3 location_object) const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();

    if (location_object.x < grid0.x + width() &&
        location_object.x >= grid0.x &&
        location_object.y < grid0.y + height() &&
        location_object.y >= grid0.y &&
        location_object.z < grid0.z + depth() &&
        location_object.z >= grid0.z) {
        return true;
    }

    return false;
}

void WiredGrid::ResetGrid() noexcept
{
    lock();
    for (auto pi = 0; pi < width_boxes_count(); pi++) {
        for (auto pj = 0; pj < height_boxes_count(); pj++) {
            for (auto pk = 0; pk < depth_boxes_count(); pk++) {
                boxes_[pi][pj][pk]->clear();
            }
        }
    }
    unlock();
}

void WiredGrid::DeleteGrid() noexcept
{
    if (boxes_ != nullptr) {
        for (auto i = 0; i < width_boxes_count(); i++) {
            for (auto j = 0; j < height_boxes_count(); j++) {
                delete [] boxes_[i][j];
            }
            delete [] boxes_[i];
        }
        delete [] boxes_;
    }
}

} // namespace grid

} // namespace nextfloor
