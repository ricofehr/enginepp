/**
 *  @file grid.cc
 *  @brief Grid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/grid.h"

#include <tbb/tbb.h>
#include <iostream>

#include "nextfloor/grid/grid_box.h"
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


Grid::Grid(EngineObject* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension)
{
    owner_ = owner;
    boxes_count_ = boxes_count;
    box_dimension_ = box_dimension;
}


void Grid::InitGrid() noexcept
{
    InitBoxes();
}

void Grid::InitBoxes() noexcept
{
    boxes_ = new std::unique_ptr<EngineGridBox> **[width_boxes_count()];

    /* Ensure pointer is allocated */
    assert(sizeof(boxes_) == sizeof(void***));

    for (auto i = 0; i < width_boxes_count(); i++) {
        boxes_[i] = new std::unique_ptr<EngineGridBox> *[height_boxes_count()];
        /* Ensure pointer is allocated */
        assert(sizeof(boxes_[i]) == sizeof(void**));

        for (auto j = 0; j < height_boxes_count(); j++) {
            boxes_[i][j] = new std::unique_ptr<EngineGridBox>[depth_boxes_count()]();
            /* Ensure pointer is allocated */
            assert(sizeof(boxes_[i][j]) == sizeof(void*));

            /* Ensure each vector is allocated */
            for (auto k = 0; k < depth_boxes_count(); k++) {
                boxes_[i][j][k] = AllocateGridBox(glm::ivec3(i,j,k));
                assert(sizeof(boxes_[i][j][k]) == sizeof(std::unique_ptr<EngineGridBox>));
            }
        }
    }
}

glm::vec3 Grid::CalculateFirstPointInGrid() const noexcept
{
    return owner_->location() - glm::vec3(width()/2, height()/2, depth()/2);
}

glm::vec3 Grid::CalculateAbsoluteCoordinates(glm::ivec3 coords) const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    auto dimension = box_dimension();

    return grid0 + glm::vec3(coords.x*dimension.x, coords.y*dimension.y, coords.z*dimension.z);
}

bool Grid::IsPositionEmpty(glm::ivec3 coords) const noexcept
{
    return boxes_[coords.x][coords.y][coords.z]->IsEmpty();
}

bool Grid::IsFrontPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.z != 0 && IsPositionFilled(glm::ivec3(coords.x,coords.y,coords.z-1));
}

bool Grid::IsRightPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.x != (width_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x+1,coords.y,coords.z));
}

bool Grid::IsLeftPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.x != 0 && IsPositionFilled(glm::ivec3(coords.x-1,coords.y,coords.z));
}

bool Grid::IsBackPositionFilled(glm::ivec3 coords) const noexcept
{
    return (coords.z != depth_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x,coords.y,coords.z+1));
}

bool Grid::IsFloorPositionFilled(glm::ivec3 coords) const noexcept
{
    return coords.y != 0 && IsPositionFilled(glm::ivec3(coords.x,coords.y-1,coords.z));
}

bool Grid::IsRoofPositionFilled(glm::ivec3 coords) const noexcept
{
    return (coords.y != height_boxes_count() - 1) && IsPositionFilled(glm::ivec3(coords.x,coords.y+1,coords.z));
}

bool Grid::IsPositionFilled(glm::ivec3 coords) const noexcept
{
    return boxes_[coords.x][coords.y][coords.z]->IsFilled();
}

std::vector<EngineGridBox*> Grid::AddItemToGrid(EngineObject* object) noexcept
{
    std::vector<glm::vec3> points = object->getCoordsModelMatrixComputed();

    auto point_min = GetMinPointFromPoints(points);
    auto point_max = GetMaxPointFromPoints(points);
    auto lengths = CalculateLengthIndexes(point_min, point_max);

    return ParseGridForObjectPlacements(object, point_min, lengths);
}

glm::ivec3 Grid::CalculateLengthIndexes(glm::vec3 point_min, glm::vec3 point_max)
{
    int length_x = static_cast<int>(ceil((point_max.x - point_min.x) / box_dimension_.x));
    int length_y = static_cast<int>(ceil((point_max.y - point_min.y) / box_dimension_.y));
    int length_z = static_cast<int>(ceil((point_max.z - point_min.z) / box_dimension_.z));

    return glm::ivec3(length_x, length_y, length_z);
}

std::vector<EngineGridBox*> Grid::ParseGridForObjectPlacements(EngineObject *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept
{
    std::vector<EngineGridBox*> coords_list;

    tbb::parallel_for (0, lengths.x, 1, [&](int cnt_x) {
        tbb::parallel_for (0, lengths.y, 1, [&](int cnt_y) {
            tbb::parallel_for (0, lengths.z, 1, [&](int cnt_z) {
                auto cnt = glm::vec3(cnt_x, cnt_y, cnt_z);
                auto intermediary_point = point_min + cnt * box_dimension_;
                auto coords = PointToGridIndexes(intermediary_point);

                if (IsCooordsAreCorrect(coords)) {
                    coords_list.push_back(AddItemToGrid(coords, object));
                }
            });
        });
    });

    return coords_list;
}

glm::ivec3 Grid::PointToGridIndexes(glm::vec3 point) noexcept
{
    glm::vec3 grid0 = CalculateFirstPointInGrid();
    glm::vec3 indexes = (glm::vec3(point.x, point.y, point.z) - grid0) / box_dimension_;

    return glm::ivec3(static_cast<int>(floor(indexes.x)), static_cast<int>(floor(indexes.y)), static_cast<int>(floor(indexes.z)));
}

EngineGridBox* Grid::AddItemToGrid(glm::ivec3 coords, EngineObject* object) noexcept
{
    assert(IsCooordsAreCorrect(coords));

    lock();
    boxes_[coords.x][coords.y][coords.z]->add(object);
    unlock();

    return boxes_[coords.x][coords.y][coords.z].get();
}

bool Grid::IsCooordsAreCorrect(glm::ivec3 coords)
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

void Grid::RemoveItemToGrid(EngineObject* object) noexcept
{
    for (auto i = 0; i < width_boxes_count(); i++) {
        for (auto j = 0; j < height_boxes_count(); j++) {
            for (auto k = 0; k < depth_boxes_count(); k++) {
                RemoveItemToGrid(glm::ivec3(i,j,k), object);
            }
        }
    }
}

void Grid::RemoveItemToGrid(glm::ivec3 coords, EngineObject* object) noexcept
{
    boxes_[coords.x][coords.y][coords.z]->remove(object);
}

void Grid::DisplayGrid() const noexcept
{
    std::string object_type{"MODEL3D"};

    if (owner_->IsUniverse()) {
        object_type = "UNIVERSE";
    } else if (owner_->IsRoom()) {
        object_type = "ROOM";
    }

    std::cout << "=== GRID FOR " << object_type << " ID " << owner_->id() << " ===" << std::endl << std::endl;
    for (auto y = 0; y < height_boxes_count(); y++) {
        std::cout << "=== Floor " << y << std::endl;
        for (auto z = 0; z < depth_boxes_count(); z++) {
            for (auto x = 0; x < width_boxes_count(); x++) {
                std::cout << "  " << boxes_[x][y][z]->size();
                // if (boxes_[x][y][z]->IsEmpty()) {
                //     std::cout << "  x";
                // } else {
                //     std::cout << "  o";
                // }
            }

            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

void Grid::ComputePlacementsInGrid() noexcept
{
    std::cout << "ComputePlacements" << std::endl;
}

glm::vec3 Grid::CalculateFrontSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 Grid::CalculateRightSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width() - 0.25f, height()/2, depth()/2);
}

glm::vec3 Grid::CalculateBackSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height()/2, depth() - 0.25f);
}

glm::vec3 Grid::CalculateLeftSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 Grid::CalculateFloorSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, 0.25f, depth()/2);
}

glm::vec3 Grid::CalculateRoofSideLocation() const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();
    return grid0 + glm::vec3(width()/2, height() - 0.25f, depth()/2);
}

glm::vec3 Grid::CalculateFrontSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 Grid::CalculateRightSideBorderScale() const noexcept
{
    return glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 Grid::CalculateBackSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, height()/2, 0.25f);
}

glm::vec3 Grid::CalculateLeftSideBorderScale() const noexcept
{
    return glm::vec3(0.25f, height()/2, depth()/2);
}

glm::vec3 Grid::CalculateFloorSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, 0.25f, depth()/2);
}

glm::vec3 Grid::CalculateRoofSideBorderScale() const noexcept
{
    return glm::vec3(width()/2, 0.25f, depth()/2);
}

bool Grid::IsInside(glm::vec3 location_object) const noexcept
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

void Grid::ResetGrid() noexcept
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

void Grid::DeleteGrid() noexcept
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
