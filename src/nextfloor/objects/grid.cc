/**
 *  @file grid.cc
 *  @brief Grid class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/grid.h"

#include <tbb/tbb.h>
#include <iostream>

#include "nextfloor/objects/grid_box.h"

namespace nextfloor {

namespace objects {

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

glm::ivec3 CalculateLengthIndexes(glm::vec3 point_min, glm::vec3 point_max, glm::vec3 grid_units)
{
    int length_x = static_cast<int>(ceil((point_max.x - point_min.x) / grid_units.x));
    int length_y = static_cast<int>(ceil((point_max.y - point_min.y) / grid_units.y));
    int length_z = static_cast<int>(ceil((point_max.z - point_min.z) / grid_units.z));

    return glm::ivec3(length_x, length_y, length_z);
}

}

Grid::Grid(EngineObject* owner)
{
    owner_ = owner;
}


void Grid::InitGrid() noexcept
{
    InitBoxes();
    ComputeGridUnits();
}

void Grid::InitBoxes() noexcept
{
    boxes_ = new std::unique_ptr<EngineGridBox> **[count_width_boxes()];

    /* Ensure pointer is allocated */
    assert(sizeof(boxes_) == sizeof(void***));

    for (auto i = 0; i < count_width_boxes(); i++) {
        boxes_[i] = new std::unique_ptr<EngineGridBox> *[count_height_boxes()];
        /* Ensure pointer is allocated */
        assert(sizeof(boxes_[i]) == sizeof(void**));

        for (auto j = 0; j < count_height_boxes(); j++) {
            boxes_[i][j] = new std::unique_ptr<EngineGridBox>[count_depth_boxes()]();
            /* Ensure pointer is allocated */
            assert(sizeof(boxes_[i][j]) == sizeof(void*));

            /* Ensure each vector is allocated */
            for (auto k = 0; k < count_depth_boxes(); k++) {
                boxes_[i][j][k] = AllocateGridBox(glm::ivec3(i,j,k));
                assert(sizeof(boxes_[i][j][k]) == sizeof(std::unique_ptr<EngineGridBox>));
            }
        }
    }
}

std::unique_ptr<EngineGridBox> Grid::AllocateGridBox(glm::ivec3 grid_coords)
{
    return std::make_unique<GridBox>(grid_coords, this);
}

glm::vec3 Grid::CalculateFirstPointInGrid() const noexcept
{
    return owner_->location() - glm::vec3(width_magnitude()/2, height_magnitude()/2, depth_magnitude()/2);
}

void Grid::ComputeGridUnits() noexcept
{
    grid_units_ = glm::vec3(x_length_by_box(), y_length_by_box(), z_length_by_box());
}

int Grid::IsPositionInTheGridEmpty(glm::ivec3 box_coords_in_grid) const noexcept
{
    return boxes_[box_coords_in_grid.x][box_coords_in_grid.y][box_coords_in_grid.z]->IsEmpty();
    // if (boxes_[box_coords_in_grid.x][box_coords_in_grid.y][box_coords_in_grid.z].size() == 0) {
    //     return kGRID_EMPTY;
    // }

    // return kGRID_USED;
}

void Grid::AddItemToGrid(EngineObject* object) noexcept
{
    std::vector<glm::vec3> points = object->getCoordsModelMatrixComputed();
    // Model3D* parent_new{nullptr};

    auto point_min = GetMinPointFromPoints(points);
    auto point_max = GetMaxPointFromPoints(points);
    auto lengths = CalculateLengthIndexes(point_min, point_max, grid_units_);

    ParseGridForObjectPlacements(object, point_min, lengths);

    // /* All coords are into other parent */
    // if (parent_new != nullptr &&
    //     parent_new != parent_ &&
    //     placements_.size() == 0) {
    //     /* change parent compute */
    //     parent_new->add_child(std::move(parent_->TransfertChild(this)));
    // }

    // /* No one parent, the current object
    //    is outside World, reverse direction */
    // if (IsMoved() && placements_.size() == 0) {
    //     InverseMove();
    // }

    //AddItemToGrid(box_coords_in_grid, object);
}

void Grid::ParseGridForObjectPlacements(EngineObject *object, glm::vec3 point_min, glm::ivec3 lengths) noexcept
{
    tbb::parallel_for (0, lengths.x, 1, [&](int cnt_x) {
        tbb::parallel_for (0, lengths.y, 1, [&](int cnt_y) {
            tbb::parallel_for (0, lengths.z, 1, [&](int cnt_z) {
                //parent_new = parent_;
                auto cnt = glm::vec3(cnt_x, cnt_y, cnt_z);
                auto intermediary_point = point_min + cnt * grid_units_;

                AddItemToGrid(PointToGridIndexes(intermediary_point), object);

                /* if i,j,k are not valid, select the good side where to search */
                //auto parent_side = parent_->BeInTheRightPlace(i, j, k);

                /* Find a new parent */
                // if (parent_side != kSAME) {
                //     parent_new = parent_->FindNeighborSide(parent_side);
                // }

                /* Add placement only if a parent exist and doesnt change */
                // if (parent_new != nullptr && parent_new == parent_) {
                //     add_placement(i, j, k);
                // }
            });
        });
    });
}

glm::ivec3 Grid::PointToGridIndexes(glm::vec3 point) noexcept
{
    glm::vec3 grid0 = CalculateFirstPointInGrid();
    glm::vec3 indexes = (glm::vec3(point.x, point.y, point.z) - grid0) / grid_units_;

    return glm::ivec3(static_cast<int>(indexes.x), static_cast<int>(indexes.y), static_cast<int>(indexes.z));
}

void Grid::AddItemToGrid(glm::ivec3 box_coords_in_grid, EngineObject* object) noexcept
{
    auto x = box_coords_in_grid.x;
    auto y = box_coords_in_grid.y;
    auto z = box_coords_in_grid.z;

    lock();
    boxes_[x][y][z]->add(object);
    unlock();
}

void Grid::RemoveItemToGrid(glm::ivec3 box_coords_in_grid, EngineObject* object) noexcept
{
    auto x = box_coords_in_grid.x;
    auto y = box_coords_in_grid.y;
    auto z = box_coords_in_grid.z;

    boxes_[x][y][z]->remove(object);
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
    for (auto y = 0; y < count_height_boxes(); y++) {
        std::cout << "=== Floor " << y << std::endl;
        for (auto z = 0; z < count_depth_boxes(); z++) {
            for (auto x = 0; x < count_width_boxes(); x++) {
                if (boxes_[x][y][z]->IsEmpty()) {
                    std::cout << "  x";
                } else {
                    std::cout << "  o";
                }
            }

            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

// std::vector<EngineObject*> Grid::FindItemsInGrid(glm::ivec3 box_coords_in_grid) const noexcept
// {
//     auto x = box_coords_in_grid.x;
//     auto y = box_coords_in_grid.y;
//     auto z = box_coords_in_grid.z;

//     return boxes_[x][y][z]->occupants();
// }

void Grid::ComputePlacementsInGrid() noexcept
{
    std::cout << "ComputePlacements" << std::endl;

    // std::vector<glm::vec3> coords = border_->getCoordsModelMatrixComputed();
    // Model3D* parent_new{nullptr};

    // auto x1 = coords.at(0)[0];
    // auto y1 = coords.at(0)[1];
    // auto z1 = coords.at(0)[2];

    // auto x2 = coords.at(0)[0];
    // auto y2 = coords.at(0)[1];
    // auto z2 = coords.at(0)[2];

    // /* Find the extremities coordinates of the border */
    // for (auto &point : coords) {
    //     if (point[0] < x1) {
    //         x1 = point[0];
    //     }

    //     if (point[1] < y1) {
    //         y1 = point[1];
    //     }

    //     if (point[2] < z1) {
    //         z1 = point[2];
    //     }

    //     if (point[0] > x2) {
    //         x2 = point[0];
    //     }

    //     if (point[1] > y2) {
    //         y2 = point[1];
    //     }

    //     if (point[2] > z2) {
    //         z2 = point[2];
    //     }
    // }

    // auto grid0 = parent_->ComputeFirstPointInGrid();
    // auto grid_unit = glm::vec3(parent_->grid_unitx(), parent_->grid_unity(), parent_->grid_unitz());

    // /* Reset current position placements */
    // clear_placements();

    // int size_x = static_cast<int>(ceil((x2 - x1) / grid_unit[0]));
    // int size_y = static_cast<int>(ceil((y2 - y1) / grid_unit[1]));
    // int size_z = static_cast<int>(ceil((z2 - z1) / grid_unit[2]));

    // tbb::parallel_for (0, size_x, 1, [&](int cnt_x) {
    //     tbb::parallel_for (0, size_y, 1, [&](int cnt_y) {
    //         tbb::parallel_for (0, size_z, 1, [&](int cnt_z) {
    //             parent_new = parent_;

    //             float x = x1 + cnt_x * grid_unit[0];
    //             float y = y1 + cnt_y * grid_unit[1];
    //             float z = z1 + cnt_z * grid_unit[2];

    //             auto tmp = (glm::vec3(x, y, z) - grid0) / grid_unit;
    //             auto i = static_cast<int>(tmp[0]);
    //             auto j = static_cast<int>(tmp[1]);
    //             auto k = static_cast<int>(tmp[2]);

    //             /* if i,j,k are not valid, select the good side where to search */
    //             auto parent_side = parent_->BeInTheRightPlace(i, j, k);

    //             /* Find a new parent */
    //             if (parent_side != kSAME) {
    //                 parent_new = parent_->FindNeighborSide(parent_side);
    //             }

    //             /* Add placement only if a parent exist and doesnt change */
    //             if (parent_new != nullptr && parent_new == parent_) {
    //                 add_placement(i, j, k);
    //             }
    //         });
    //     });
    // });

    // /* All coords are into other parent */
    // if (parent_new != nullptr &&
    //     parent_new != parent_ &&
    //     placements_.size() == 0) {
    //     /* change parent compute */
    //     parent_new->add_child(std::move(parent_->TransfertChild(this)));
    // }

    // /* No one parent, the current object
    //    is outside World, reverse direction */
    // if (IsMoved() && placements_.size() == 0) {
    //     InverseMove();
    // }
}

bool Grid::IsInside(glm::vec3 location_object) const noexcept
{
    auto grid0 = CalculateFirstPointInGrid();

    if (location_object.x < grid0.x + width_magnitude() &&
        location_object.x >= grid0.x &&
        location_object.y < grid0.y + height_magnitude() &&
        location_object.y >= grid0.y &&
        location_object.z < grid0.z + depth_magnitude() &&
        location_object.z >= grid0.z) {
        return true;
    }

    return false;
}

void Grid::ResetGrid() noexcept
{
    lock();
    for (auto pi = 0; pi < count_width_boxes(); pi++) {
        for (auto pj = 0; pj < count_height_boxes(); pj++) {
            for (auto pk = 0; pk < count_depth_boxes(); pk++) {
                boxes_[pi][pj][pk]->clear();
            }
        }
    }
    unlock();
}

void Grid::DeleteGrid() noexcept
{
    std::cout << count_width_boxes() << std::endl;

    if (boxes_ != nullptr) {
        for (auto i = 0; i < count_width_boxes(); i++) {
            for (auto j = 0; j < count_height_boxes(); j++) {
                delete [] boxes_[i][j];
            }
            delete [] boxes_[i];
        }
        delete [] boxes_;
    }
}

} // namespace objects

} // namespace nextfloor
