/**
 *  @file model3d.cc
 *  @brief Model3d class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *  This class file is ordered in the following ways
 *  - Constructor and Initialization Steps
 *  - Clipping Neighbors Selection
 *  - Collision detection
 *  - Moving and Locations compute
 *  - Some tool functions for the grid
 *
 */

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cilk/cilk.h>

#include "engine/physics/cl_collision_engine.h"
#include "engine/physics/serial_collision_engine.h"
#include "engine/physics/cilk_collision_engine.h"
#include "engine/core/config_engine.h"

namespace engine {

namespace universe {

namespace {
    /* Unique id for object */
    static int sObjectId = 1;
} // anonymous namespace

Model3D::Model3D()
{
    id_ = sObjectId++;
    obstacle_ = nullptr;
    id_last_collision_ = 0;
    is_collision_with_camera_ = false;
    collision_countdown_ = 0;
    InitCollisionEngine();
}

bool operator==(const Model3D& o1, const Model3D& o2)
{
    return o1.id() == o2.id();
}

bool operator!=(const Model3D& o1, const Model3D& o2)
{
    return o1.id() != o2.id();
}

void Model3D::InitCollisionEngine()
{
    using engine::physics::CollisionEngine;
    using engine::core::ConfigEngine;

    /* Get parallell type from config */
    int type_parallell = ConfigEngine::getSetting<int>("parallell");

    switch (type_parallell) {
        case CollisionEngine::kPARALLELL_CILK:
            collision_engine_ = engine::physics::CilkCollisionEngine::Instance();
            break;
        case CollisionEngine::kPARALLELL_CL:
            collision_engine_ = engine::physics::CLCollisionEngine::Instance();
            break;
        default:
            collision_engine_ = engine::physics::SerialCollisionEngine::Instance();
            break;
    }

    assert(collision_engine_ != nullptr);
}

void Model3D::InitGrid()
{
    grid_ = new std::vector<Model3D*> **[grid_x_];
    /* Ensure pointer is allocated */
    assert(sizeof(grid_) == sizeof(void***));

    for (auto i = 0; i < grid_x_; i++) {
        grid_[i] = new std::vector<Model3D*> *[grid_y_];
        /* Ensure pointer is allocated */
        assert(sizeof(grid_[i]) == sizeof(void**));

        for (auto j = 0; j < grid_y_; j++) {
            grid_[i][j] = new std::vector<Model3D*>[grid_z_]();
            /* Ensure pointer is allocated */
            assert(sizeof(grid_[i][j]) == sizeof(void*));

            /* Ensure each vector is allocated */
            for (auto k = 0; k < grid_z_; k++) {
                assert(sizeof(grid_[i][j][k]) == sizeof(std::vector<Model3D*>));
            }
        }
    }
}

std::vector<Model3D*> Model3D::FindClippingNeighbors(int level) const noexcept
{
    std::vector<Model3D*> ret(0);
    auto neighs = FindAllNeighbors();
    auto cnt = level;

    /* Filter neighbor of current Model3D */
    for (auto &neighbor : neighs) {
        if (IsClippingNear(neighbor, level)) {
            if (neighbor != this && std::find(ret.begin(), ret.end(), neighbor) == ret.end()) {
                ret.push_back(neighbor);
            }
        }
    }

    /* Add neighbors of neighbors in deeper level */
    while (cnt-- != 0) {
        /* Copy ret for avoid changing this one during parsing these */
        auto tmp = ret;
        for (auto &o : tmp) {
            for (auto &neighbor : o->FindAllNeighbors()) {
                if (IsClippingNear(neighbor, level)) {
                    if (neighbor != this && std::find(ret.begin(), ret.end(), neighbor) == ret.end()) {
                        ret.push_back(neighbor);
                    }
                }
            }
        }
    }

    return ret;
}

std::vector<Model3D*> Model3D::FindAllNeighbors() const noexcept
{
    std::vector<Model3D*> ret(0);
    tbb::mutex neighbor_mutex;

    cilk_for (auto cnp = 0; cnp < placements_.size(); cnp++) {
        auto i = placements_[cnp][0];
        auto j = placements_[cnp][1];
        auto k = placements_[cnp][2];

        cilk_for (auto cnt = 0; cnt < kSIDES; cnt++) {
            auto c = GetNeighborCoordsBySide(i, j, k, cnt);

            for (auto &neighbor : parent_->FindItemsInGrid(c[0], c[1], c[2])) {
                if (neighbor != this) {
                    tbb::mutex::scoped_lock lock_map(neighbor_mutex);
                    if (std::find(ret.begin(), ret.end(), neighbor) == ret.end()) {
                        ret.push_back(neighbor);
                    }
                }
            }
        }
    }

    return ret;
}

std::vector<Model3D*> Model3D::FindItemsInGrid(int i, int j, int k) const noexcept
{
    std::vector<Model3D*> empty_ret(0);

    if (i < 0) {
        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0]-1, placements_[0][1], placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(neighbor[0]->gridx()-1, j, k);
        } else {
            return empty_ret;
        }
    }

    if (i == grid_x_) {
        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0]+1, placements_[0][1], placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(0, j, k);
        } else {
            return empty_ret;
        }
    }

    if (j < 0) {
        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1]-1, placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(i, neighbor[0]->gridy()-1, k);
        } else {
            return empty_ret;
        }
    }

    if (j == grid_y_) {
        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1]+1, placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(i, 0, k);
        } else {
            return empty_ret;
        }
    }

    if (k < 0) {
        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1], placements_[0][2]-1);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(i, j, neighbor[0]->gridz()-1);
        } else {
            return empty_ret;
        }
    }

    if (k == grid_z_) {

        if (parent_ == nullptr) {
            return empty_ret;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1], placements_[0][2]+1);

        if (neighbor.size() > 0) {
            return neighbor[0]->FindItemsInGrid(i, j, 0);
        } else {
            return empty_ret;
        }
    }

    return grid_[i][j][k];
}

bool Model3D::IsClippingNear(Model3D* target, int level) const noexcept
{
    for (auto &p1 : placements_) {
        for (auto &p2 : target->placements()) {
            /* Hide object if noone coord is same */
            if (p1[0] != p2[0] &&
                p1[1] != p2[1] &&
                p1[2] != p2[2]) {
                continue;
            }

            /* Display any object who is just next */
            if (abs(p1[0] - p2[0]) <= 1 &&
                abs(p1[1] - p2[1]) <= 1 &&
                abs(p1[2] - p2[2]) <= 1) {
                return true;
            }

            /* Hide object which dont have 2 coords in common */
            if ((p1[0] != p2[0] &&
                 p1[1] != p2[1]) ||
                (p1[0] != p2[0] &&
                 p1[2] != p2[2]) ||
                (p1[1] != p2[1] &&
                 p1[2] != p2[2])) {
                    continue;
                }

            /* Display objects if level of deep is good */
            if (abs(p1[0] - p2[0]) <= level &&
                abs(p1[1] - p2[1]) <= level &&
                abs(p1[2] - p2[2]) <= level) {
                return true;
            }
        }
    }

    return false;
}

void Model3D::DetectCollision() noexcept
{
    if (IsMoved()) {
        PivotCollision();
    }

    cilk_for (auto i = 0; i < objects_.size(); i++) {
        objects_[i]->DetectCollision();
    }
}

void Model3D::PivotCollision() noexcept
{
    /* Prepare vector for collision compute */
    std::vector<Model3D*> test_objects = FindCollisionNeighbors();

    /* Parallell collision loop for objects with cilkplus */
    cilk_for (auto i = 0; i < test_objects.size(); i++) {
        /* Abort program if object and room_object loop are same (must no happend) */
        assert(*this != *test_objects[i]);

        collision_engine_->DetectCollision(this, test_objects[i]);
    }

    if (IsCamera() && obstacle() != nullptr) {
        obstacle()->flag_collision_with_camera();
    }
}

std::vector<Model3D*> Model3D::FindCollisionNeighbors() const noexcept
{
    std::vector<Model3D*> ret(0);
    tbb::mutex neighbor_mutex;

    auto dirx = IsMovedX();
    auto diry = IsMovedY();
    auto dirz = IsMovedZ();

    auto sides = parent_->ListSidesInTheDirection(dirx, diry, dirz);
    cilk_for(auto cnt = 0; cnt < sides.size(); cnt++) {
        for (auto &neighbor: FindNeighborsSide(sides[cnt])) {

            /* Avoid add last collision objects between both */
            if (neighbor->id() == id_last_collision() ||
                id() == neighbor->id_last_collision()) {
                continue;
            }

            tbb::mutex::scoped_lock lock_map(neighbor_mutex);
            if(std::find(ret.begin(), ret.end(), neighbor) == ret.end()) {
                ret.push_back(neighbor);
            }
        }
    }

    return ret;
}

const std::vector<int> Model3D::ListSidesInTheDirection(int dirx, int diry, int dirz) const noexcept
{
    /* Left sides */
    if (dirx == -1 && diry == -1 && dirz == -1) {
        return {kSAME, kLEFT, kLEFT_FLOOR, kFLOOR, kLEFT_BACK, kLEFT_FLOOR_BACK, kFLOOR_BACK, kBACK};
    }

    if (dirx == -1 && diry == -1 && dirz == 1) {
        return {kSAME, kLEFT, kLEFT_FLOOR, kFLOOR, kLEFT_FRONT, kLEFT_FLOOR_FRONT, kFLOOR_FRONT, kFRONT};
    }

    if (dirx == -1 && diry == -1) {
        return {kSAME, kLEFT, kLEFT_FLOOR, kFLOOR};
    }

    if (dirx == -1 && diry == 1 && dirz == -1) {
        return {kSAME, kLEFT, kLEFT_ROOF, kROOF, kLEFT_BACK, kLEFT_ROOF_BACK, kROOF_BACK, kBACK};
    }

    if (dirx == -1 && diry == 1 && dirz == 1) {
        return {kSAME, kLEFT, kLEFT_ROOF, kROOF, kLEFT_FRONT, kLEFT_ROOF_FRONT, kROOF_FRONT, kFRONT};
    }

    if (dirx == -1 && diry == 1) {
        return {kSAME, kLEFT, kLEFT_ROOF, kROOF};
    }

    if (dirx == -1 && dirz == -1) {
        return {kSAME, kLEFT, kLEFT_BACK, kBACK};
    }

    if (dirx == -1 && dirz == 1) {
        return {kSAME, kLEFT, kLEFT_FRONT, kFRONT};
    }

    if (dirx == -1) {
        return {kSAME, kLEFT};
    }

    /* Right sides */
    if (dirx == 1 && diry == -1 && dirz == -1) {
        return {kSAME, kRIGHT, kRIGHT_FLOOR, kFLOOR, kRIGHT_BACK, kRIGHT_FLOOR_BACK, kFLOOR_BACK, kBACK};
    }

    if (dirx == 1 && diry == -1 && dirz == 1) {
        return {kSAME, kRIGHT, kRIGHT_FLOOR, kFLOOR, kRIGHT_FRONT, kRIGHT_FLOOR_FRONT, kFLOOR_FRONT, kFRONT};
    }

    if (dirx == 1 && diry == -1) {
        return {kSAME, kRIGHT, kRIGHT_FLOOR, kFLOOR};
    }

    if (dirx == 1 && diry == 1 && dirz == -1) {
        return {kSAME, kRIGHT, kRIGHT_ROOF, kROOF, kRIGHT_BACK, kRIGHT_ROOF_BACK, kROOF_BACK, kBACK};
    }

    if (dirx == 1 && diry == 1 && dirz == 1) {
        return {kSAME, kRIGHT, kRIGHT_ROOF, kROOF, kRIGHT_FRONT, kRIGHT_ROOF_FRONT, kROOF_FRONT, kFRONT};
    }

    if (dirx == 1 && diry == 1) {
        return {kSAME, kRIGHT, kRIGHT_ROOF, kROOF};
    }

    if (dirx == 1 && dirz == -1) {
        return {kSAME, kRIGHT, kRIGHT_BACK, kBACK};
    }

    if (dirx == 1 && dirz == 1) {
        return {kSAME, kRIGHT, kRIGHT_FRONT, kFRONT};
    }

    if (dirx == 1) {
        return {kSAME, kRIGHT};
    }

    /* Floor sides */
    if (diry == -1 && dirz == -1) {
        return {kSAME, kFLOOR, kFLOOR_BACK, kBACK};
    }

    if (diry == -1 && dirz == 1) {
        return {kSAME, kFLOOR, kFLOOR_FRONT, kFRONT};
    }

    if (diry == -1) {
        return {kSAME, kFLOOR};
    }

    /* Roof sides */
    if (diry == 1 && dirz == -1) {
        return {kSAME, kROOF, kROOF_BACK, kBACK};
    }

    if (diry == 1 && dirz == 1) {
        return {kSAME, kROOF, kROOF_FRONT, kFRONT};
    }

    if (diry == 1) {
        return {kSAME, kROOF};
    }

    /* Back sides */
    if (dirz == -1) {
        return {kSAME, kBACK};
    }

    /* Front sides */
    if (dirz == 1) {
        return {kSAME, kFRONT};
    }

    /* Must not to be here */
    return {kSAME};
}

std::vector<Model3D*> Model3D::FindNeighborsSide(int side) const noexcept
{
    std::vector<Model3D*> ret(0);
    tbb::mutex neighbor_mutex;

    cilk_for (auto cnp = 0; cnp < placements_.size(); cnp++) {
        auto i = placements_[cnp][0];
        auto j = placements_[cnp][1];
        auto k = placements_[cnp][2];
        auto c = GetNeighborCoordsBySide(i, j, k, side);

        for (auto &neighbor : parent_->FindItemsInGrid(c[0], c[1], c[2])) {
            if (neighbor != this) {
                tbb::mutex::scoped_lock lock_map(neighbor_mutex);
                if (std::find(ret.begin(), ret.end(), neighbor) == ret.end()) {
                    ret.push_back(neighbor);
                }
            }
        }
    }

    return ret;
}

glm::vec3 Model3D::GetNeighborCoordsBySide(int i, int j, int k, int side) const
{
    switch(side) {
        case kSAME: return glm::vec3(i,j,k);
        case kLEFT_FLOOR_BACK: return glm::vec3(i-1,j-1,k-1);
        case kLEFT_FLOOR_FRONT: return glm::vec3(i-1,j-1,k+1);
        case kLEFT_FLOOR: return glm::vec3(i-1,j-1,k);
        case kLEFT_ROOF_BACK: return glm::vec3(i-1,j+1,k-1);
        case kLEFT_ROOF_FRONT: return glm::vec3(i-1,j+1,k+1);
        case kLEFT_ROOF: return glm::vec3(i-1,j+1,k);
        case kLEFT_BACK: return glm::vec3(i-1,j,k-1);
        case kLEFT_FRONT: return glm::vec3(i-1,j,k+1);
        case kLEFT: return glm::vec3(i-1,j,k);
        case kRIGHT_FLOOR_BACK: return glm::vec3(i+1,j-1,k-1);
        case kRIGHT_FLOOR_FRONT: return glm::vec3(i+1,j-1,k+1);
        case kRIGHT_FLOOR: return glm::vec3(i+1,j-1,k);
        case kRIGHT_ROOF_BACK: return glm::vec3(i+1,j+1,k-1);
        case kRIGHT_ROOF_FRONT: return glm::vec3(i+1,j+1,k+1);
        case kRIGHT_ROOF: return glm::vec3(i+1,j+1,k);
        case kRIGHT_BACK: return glm::vec3(i+1,j,k-1);
        case kRIGHT_FRONT: return glm::vec3(i+1,j,k+1);
        case kRIGHT: return glm::vec3(i+1,j,k);
        case kFLOOR_BACK: return glm::vec3(i,j-1,k-1);
        case kFLOOR_FRONT: return glm::vec3(i,j-1,k+1);
        case kFLOOR: return glm::vec3(i,j-1,k);
        case kROOF_BACK: return glm::vec3(i,j+1,k-1);
        case kROOF_FRONT: return glm::vec3(i,j+1,k+1);
        case kROOF: return glm::vec3(i,j+1,k);
        case kBACK: return glm::vec3(i,j,k-1);
        case kFRONT: return glm::vec3(i,j,k+1);
    }

    return glm::vec3(i,j,k);
}

void Model3D::Move() noexcept
{
    border_->MoveCoords();

    cilk_for (auto cnt = 0; cnt < elements_.size(); cnt++) {
        elements_[cnt]->ComputeMVP();
    }

    if (collision_countdown_ == 0) {
        id_last_collision_ = 0;
    } else {
        --collision_countdown_;
    }

    /* Ensure next kCOLLISION_COUNTDOWN frames
       dont compute collision with same obstacle */
    if (obstacle_ != nullptr && !IsCameraCollision()) {
        id_last_collision_ = obstacle_->id();
        collision_countdown_ = kCOLLISION_COUNTDOWN;
    }

    if (collision_countdown_ == 0) {
        id_last_collision_ = 0;
    } else {
        --collision_countdown_;
    }

    obstacle_ = nullptr;
    is_collision_with_camera_ = false;

    /* New placements in parent grid */
    if (IsMoved()) {
        ComputePlacements();
    }

    if (countChilds() == 0) {
        return;
    }

    /*
     *  Compute GL coords for childs
     *  ComputePlacements (called by Move) can move child to another parent
     *  So, need to have a sequential loop
     */
    for (auto cnt = 0; cnt < objects_.size();) {
        auto nb_objs = objects_.size();

        objects_[cnt]->Move();

        if (nb_objs == objects_.size()) {
            cnt++;
        }
    }
}

void Model3D::ComputePlacements() noexcept
{
    /* If no parent, no position placements */
    if (parent_ == nullptr) {
        return;
    }

    std::vector<glm::vec3> coords = border_->ComputeCoords();
    Model3D* parent_new{nullptr};

    auto x1 = coords.at(0)[0];
    auto y1 = coords.at(0)[1];
    auto z1 = coords.at(0)[2];

    auto x2 = coords.at(0)[0];
    auto y2 = coords.at(0)[1];
    auto z2 = coords.at(0)[2];

    /* Find the extremities coordinates of the border */
    for (auto &point : coords) {
        if (point[0] < x1) {
            x1 = point[0];
        }

        if (point[1] < y1) {
            y1 = point[1];
        }

        if (point[2] < z1) {
            z1 = point[2];
        }

        if (point[0] > x2) {
            x2 = point[0];
        }

        if (point[1] > y2) {
            y2 = point[1];
        }

        if (point[2] > z2) {
            z2 = point[2];
        }
    }

    auto grid0 = parent_->GetGrid0();
    auto grid_unit = glm::vec3(parent_->grid_unitx(), parent_->grid_unity(), parent_->grid_unitz());

    /* Reset current position placements */
    clear_placements();

    int size_x = static_cast<int>(ceil((x2 - x1) / grid_unit[0]));
    int size_y = static_cast<int>(ceil((y2 - y1) / grid_unit[1]));
    int size_z = static_cast<int>(ceil((z2 - z1) / grid_unit[2]));

    cilk_for (auto cnt_x = 0 ; cnt_x < size_x ; cnt_x++) {
        cilk_for (auto cnt_y = 0 ; cnt_y < size_y ; cnt_y++) {
            cilk_for (auto cnt_z = 0 ; cnt_z < size_z ; cnt_z++) {
                parent_new = parent_;

                float x = x1 + cnt_x * grid_unit[0];
                float y = y1 + cnt_y * grid_unit[1];
                float z = z1 + cnt_z * grid_unit[2];

                auto tmp = (glm::vec3(x, y, z) - grid0) / grid_unit;
                auto i = static_cast<int>(tmp[0]);
                auto j = static_cast<int>(tmp[1]);
                auto k = static_cast<int>(tmp[2]);

                /* if i,j,k are not valid, select the good side where to search */
                auto parent_side = parent_->BeInTheRightPlace(i, j, k);

                /* Find a new parent */
                if (parent_side != kSAME) {
                    parent_new = parent_->FindNeighborSide(parent_side);
                }

                /* If no parent for current coords */
                if (parent_new == nullptr) {
                    continue;
                }

                /* If parent change for current coords */
                if (parent_new != parent_) {
                    continue;
                }

                add_placement(i, j, k);
            }
        }
    }

    /* All coords are into other parent */
    if (parent_new != nullptr &&
        parent_new != parent_ &&
        placements_.size() == 0) {
        /* change parent compute */
        parent_new->add_child(std::move(parent_->TransfertChild(this)));
    }

    /* No one parent, the current object
       is outside World, reverse direction */
    if (IsMoved() && placements_.size() == 0) {
        InverseMove();
    }
}

int Model3D::BeInTheRightPlace(int i, int j, int k) const
{
    if (i < 0 && j < 0 && k < 0) {
        return kLEFT_FLOOR_BACK;
    }

    if (i < 0 && j < 0 && k >= gridz()) {
        return kLEFT_FLOOR_FRONT;
    }

    if (i < 0 && j < 0) {
        return kLEFT_FLOOR;
    }

    if (i < 0 && j >= gridy() && k < 0) {
        return kLEFT_ROOF_BACK;
    }

    if (i < 0 && j >= gridy() && k >= gridz()) {
        return kLEFT_ROOF_FRONT;
    }

    if (i < 0 && j >= gridy()) {
        return kLEFT_ROOF;
    }

    if (i < 0 && k < 0) {
        return kLEFT_BACK;
    }

    if (i < 0 && k >= gridz()) {
        return kLEFT_FRONT;
    }

    if (i < 0) {
        return kLEFT;
    }

    if (i >= gridx() && j < 0 && k < 0) {
        return kRIGHT_FLOOR_BACK;
    }

    if (i >= gridx() && j < 0 && k >= gridz()) {
        return kRIGHT_FLOOR_FRONT;
    }

    if (i >= gridx() && j < 0) {
        return kRIGHT_FLOOR;
    }

    if (i >= gridx() && j >= gridy() && k < 0) {
        return kRIGHT_ROOF_BACK;
    }

    if (i >= gridx() && j >= gridy() && k >= gridz()) {
        return kRIGHT_ROOF_FRONT;
    }

    if (i >= gridx() && j >= gridy()) {
        return kRIGHT_ROOF;
    }

    if (i >= gridx() && k < 0) {
        return kRIGHT_BACK;
    }

    if (i >= gridx() && k >= gridz()) {
        return kRIGHT_FRONT;
    }

    if (i >= gridx()) {
        return kRIGHT;
    }

    if (j < 0 && k < 0) {
        return kFLOOR_BACK;
    }

    if (j < 0 && k >= gridz()) {
        return kFLOOR_FRONT;
    }

    if (j < 0) {
        return kFLOOR;
    }

    if (j >= gridy() && k < 0) {
        return kROOF_BACK;
    }

    if (j >= gridy() && k >= gridz()) {
        return kROOF_FRONT;
    }

    if (j >= gridy()) {
        return kROOF;
    }

    if (k < 0) {
        return kBACK;
    }

    if (k >= gridz()) {
        return kFRONT;
    }

    /* kSAME -> already the right place */
    return kSAME;
}

Model3D* Model3D::FindNeighborSide(int side) const noexcept
{
    for (auto &p: placements_) {
        auto i = p[0];
        auto j = p[1];
        auto k = p[2];
        auto c = GetNeighborCoordsBySide(i, j, k, side);

        for (auto &neighbor : parent_->FindItemsInGrid(c[0], c[1], c[2])) {
            if (neighbor != this) {
                return neighbor;
            }
        }
    }

    return nullptr;
}

std::unique_ptr<Model3D> Model3D::TransfertChild(Model3D* child) noexcept
{
    std::unique_ptr<Model3D> ret{nullptr};

    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (objects_[cnt].get() == child) {
            lock();
            auto initial_count_childs = objects_.size();
            ret = std::move(objects_[cnt]);
            objects_.erase(objects_.begin() + cnt);
            /* Ensure child is erased from current objects_ array */
            assert(initial_count_childs == objects_.size() + 1);
            unlock();

            return ret;
        }
    }

    return ret;
}

int Model3D::IsPositionInTheGridEmpty(int i, int j, int k) const noexcept
{
    if (i < 0) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0]-1, placements_[0][1], placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(neighbor[0]->gridx()-1, j, k);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (i == grid_x_) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0]+1, placements_[0][1], placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(0, j, k);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (j < 0) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1]-1, placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(i, neighbor[0]->gridy()-1, k);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (j == grid_y_) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1]+1, placements_[0][2]);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(i, 0, k);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (k < 0) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1], placements_[0][2]-1);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(i, j, neighbor[0]->gridz()-1);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (k == grid_z_) {
        if (parent_ == nullptr) {
            return kGRID_UNKNOW;
        }

        auto neighbor = parent_->FindItemsInGrid(placements_[0][0], placements_[0][1], placements_[0][2]+1);

        if (neighbor.size() > 0) {
            return neighbor[0]->IsPositionInTheGridEmpty(i, j, 0);
        } else {
            return kGRID_UNKNOW;
        }
    }

    if (grid_[i][j][k].size() == 0) {
        return kGRID_EMPTY;
    }

    return kGRID_USED;
}

void Model3D::AddItemToGrid(int i, int j, int k, Model3D* obj) noexcept
{
    lock();
    grid_[i][j][k].push_back(obj);
    unlock();
}

void Model3D::RemoveItemToGrid(int i, int j, int k, Model3D* obj) noexcept
{
    for (auto cnt = 0; cnt < grid_[i][j][k].size(); cnt++) {
        if (grid_[i][j][k][cnt] == obj) {
            lock();
            auto initial_count_items = grid_[i][j][k].size();
            grid_[i][j][k].erase(grid_[i][j][k].begin() + cnt);
            /* Ensure item is erased from grid position */
            assert(initial_count_items == grid_[i][j][k].size() + 1);
            unlock();
            return;
        }
    }
}

void Model3D::DisplayGrid() const noexcept
{
    std::string object_type{"MODEL3D"};

    if (IsUniverse()) {
        object_type = "UNIVERSE";
    } else if (IsRoom()) {
        object_type = "ROOM";
    }

    std::cout << "=== GRID FOR " << object_type << " ID " << id_ << " ===" << std::endl << std::endl;
    for (auto i = 0; i < grid_y_; i++) {
        std::cout << "=== Floor " << i << std::endl;
        for (auto k = 0; k < grid_z_; k++) {
            for (auto j = 0; j < grid_x_; j++) {
                if (grid_[j][i][k].size() > 0) {
                    std::cout << "  o";
                } else {
                    std::cout << "  x";
                }
            }

            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

bool Model3D::IsInside(glm::vec3 location_object) const
{
    auto location_0 = location() - glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2);

    if (location_object[0] < location_0[0] + grid_x_ * grid_unit_x_ &&
        location_object[0] >= location_0[0] &&
        location_object[1] < location_0[1] + grid_y_ * grid_unit_y_ &&
        location_object[1] >= location_0[1] &&
        location_object[2] < location_0[2] + grid_z_ * grid_unit_z_ &&
        location_object[2] >= location_0[2]) {
        return true;
    }

    return false;
}

void Model3D::ResetGrid() noexcept
{
    lock();
    for (auto pi = 0; pi < grid_x_; pi++) {
        for (auto pj = 0; pj < grid_y_; pj++) {
            for (auto pk = 0; pk < grid_z_; pk++) {
                grid_[pi][pj][pk].clear();
            }
        }
    }
    unlock();
}

Model3D::~Model3D()
{
    if (grid_ != nullptr) {
        for (auto i = 0; i < grid_x_; i++) {
            for (auto j = 0; j < grid_y_; j++) {
                delete [] grid_[i][j];
            }
            delete [] grid_[i];
        }
        delete [] grid_;
    }
}

} // namespace universe

} // namespace engine
