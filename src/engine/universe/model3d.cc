/*
 *   Model3d class file
 *   @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

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
    distance_ = -1.0f;
    obstacle_ = nullptr;
    id_last_collision_ = 0;
    is_controlled_ = false;
    is_crossed_ = false;
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
}

void Model3D::Move() noexcept
{
    border_->set_distance(distance_);
    border_->MoveCoords();
    for (auto &element : elements_) {
        element->set_distance(distance_);
        element->ComputeMVP();
    }
    distance_ = -1.0f;
    /* An object cant touch same object twice, except camera */
    id_last_collision_ = -1;
    if (!is_controlled_ && obstacle_ != nullptr) {
        id_last_collision_ = obstacle_->id();
    }
    obstacle_ = nullptr;

    if (countChilds() == 0) {
        return;
    }
    /* Compute coords for first child object. Important if this one is camera */
    objects_[0]->Move();

    /* Compute objects GL coords  */
    cilk_for (auto cnt = 1; cnt < objects_.size(); cnt++) {
        objects_[cnt]->Move();
    }
}

void Model3D::InitGrid()
{
    grid_ = new std::vector<Model3D*> **[grid_x_];
    for (auto i = 0; i < grid_x_; i++) {
        grid_[i] = new std::vector<Model3D*> *[grid_y_];
        for (auto j = 0; j < grid_y_; j++) {
            grid_[i][j] = new std::vector<Model3D*>[grid_z_]();
        }
    }
}

std::vector<std::unique_ptr<Model3D>> Model3D::ReinitGrid() noexcept
{
    std::vector<std::unique_ptr<Model3D>> ret;
    for (auto o = 0; o < objects_.size();) {

        if (!objects_[o]->IsMoved() &&
            objects_[o]->placements().size() > 0) {
            ++o;
            continue;
        }

        /* check grid collision */
        auto border = objects_[o]->border();
        std::vector<glm::vec3> coords = border->ComputeCoords();
        auto x1 = coords.at(0)[0];
        auto y1 = coords.at(0)[1];
        auto z1 = coords.at(0)[2];
        auto w1 = coords.at(1)[0] - x1;
        auto h1 = coords.at(3)[1] - y1;
        auto d1 = coords.at(4)[2] - z1;

        for (auto &p : objects_[o]->placements()) {
            tbb::mutex::scoped_lock lock(object_mutex_);

            auto pi = p[0];
            auto pj = p[1];
            auto pk = p[2];

            auto it = std::find(grid_[pi][pj][pk].begin(), grid_[pi][pj][pk].end(), objects_[o].get());
            if(it != grid_[pi][pj][pk].end()) {
                grid_[pi][pj][pk].erase(it);
            }
        }

        objects_[o]->clear_placements();
        auto grid_0 = location() - glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2);

        cilk_for (auto i = 0; i < grid_x_; i++) {
            auto x2 = grid_0[0] + i * grid_unit_x_;
            cilk_for (auto j = 0; j < grid_y_; j++) {
                auto y2 = grid_0[1] + (j+1) * grid_unit_y_;
                cilk_for (auto k = 0; k < grid_z_; k++) {
                    auto z2 = grid_0[2] + (k+1) * grid_unit_z_;
                    auto h2 = -grid_unit_y_;
                    auto w2 = grid_unit_x_;
                    auto d2 = -grid_unit_z_;

                    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
                        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                        tbb::mutex::scoped_lock lock(object_mutex_);
                        objects_[o]->add_placement(i, j, k);
                        grid_[i][j][k].push_back(objects_[o].get());
                    }
                }
            }
        }

        if (objects_[o]->placements().size() == 0) {
            tbb::mutex::scoped_lock lock(object_mutex_);
            ret.push_back(std::move(objects_[o]));
            objects_.erase(objects_.begin() + o);
        } else {
            o++;
        }
    }

    return ret;
}

void Model3D::DisplayGrid() noexcept
{
    std::string object_type{"MODEL3D"};
    if (type_ == kMODEL3D_UNIVERSE) {
        object_type = "UNIVERSE";
    }
    else if (type_ == kMODEL3D_ROOM) {
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

std::unique_ptr<Model3D> Model3D::TransfertObject(std::unique_ptr<Model3D> child, bool force) noexcept
{
    if (force || IsInside(child->location())) {
        /* Output if debug setted */
        using engine::core::ConfigEngine;
        if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_COLLISION && force) {
            std::cout << "Transfert " << child->id() << " to Room " << id() << " (forced: " << force << ")" << std::endl;
        }

        /* Inverse object move if forced transfert */
        if (force) {
            child->InverseMove();
        }

        /* If obj is Camera, insert into first child.
           Else insert in last position. */
        if (child->type() == Model3D::kMODEL3D_CAMERA) {
            objects_.insert(objects_.begin(), std::move(child));
        } else {
            objects_.push_back(std::move(child));
        }

        return nullptr;
    }

    return child;
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

std::vector<std::unique_ptr<Model3D>> Model3D::ListOutsideObjects() noexcept
{
    std::vector<std::unique_ptr<Model3D>> ret;
    for (auto &o : objects_) {
        if (!IsInside(o->location())) {
            ret.push_back(std::move(o));
        }
    }
    
    return ret;
}

void Model3D::DetectCollision(std::vector<Model3D*> neighbors) noexcept
{
    /* For all others moving objects
       Parallell loop with cilkplus */
    cilk_for (auto i = 0; i < objects_.size(); i++) {
        if (objects_[i]->IsMoved()) {
            PivotCollision(objects_[i].get(), neighbors);
        }
    }
}

void Model3D::PivotCollision(Model3D* target, std::vector<Model3D*> neighbors) noexcept
{
    tbb::mutex pivot_mutex;
    /* ensure that 2 same objects are not checked at the same time */
    target->lock();

    /* Prepare an unique objects collection for collision detection */
    std::map<int, Model3D*> grid_objects;

    /* Grid coordinates for target */
    std::vector<std::vector<int>> placements = target->placements();

    /* Check all grid placements for current object and select other objects near this one */
    cilk_for (auto p = 0; p < placements.size(); p++) {
        auto x = placements[p][0];
        auto y = placements[p][1];
        auto z = placements[p][2];

        /* Other objects adding */
        cilk_for (auto i = x-1; i <= x+1; i++) {
            cilk_for (auto j = y-1; j <= y+1; j++) {
                cilk_for (auto k = z-1; k <= z+1; k++) {

                    std::vector<Model3D*> brothers;

                    if (i < 0) {
                        if (j < 0) {
                            if (k < 0) {
                                if (neighbors[kLEFT_FLOOR_FRONT] != nullptr) {
                                    brothers = neighbors[kLEFT_FLOOR_FRONT]->getPlacementObjects(grid_x_-1,
                                                                                                 grid_y_-1,
                                                                                                 grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kLEFT_FLOOR_BACK] != nullptr) {
                                    brothers = neighbors[kLEFT_FLOOR_BACK]->getPlacementObjects(grid_x_-1, grid_y_-1, 0);
                                }
                            } else {
                                if (neighbors[kLEFT_FLOOR] != nullptr) {
                                    brothers = neighbors[kLEFT_FLOOR]->getPlacementObjects(grid_x_-1, grid_y_-1, k);
                                }
                            }
                        } else if (j == grid_y_) {
                            if (k < 0) {
                                if (neighbors[kLEFT_ROOF_FRONT] != nullptr) {
                                    brothers = neighbors[kLEFT_ROOF_FRONT]->getPlacementObjects(grid_x_-1, 0, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kLEFT_ROOF_BACK] != nullptr) {
                                    brothers = neighbors[kLEFT_ROOF_BACK]->getPlacementObjects(grid_x_-1, 0, 0);
                                }
                            } else {
                                if (neighbors[kLEFT_ROOF] != nullptr) {
                                    brothers = neighbors[kLEFT_ROOF]->getPlacementObjects(grid_x_-1, 0, k);
                                }
                            }
                        } else {
                            if (k < 0) {
                                if (neighbors[kLEFT_FRONT] != nullptr) {
                                    brothers = neighbors[kLEFT_FRONT]->getPlacementObjects(grid_x_-1, j, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kLEFT_BACK] != nullptr) {
                                    brothers = neighbors[kLEFT_BACK]->getPlacementObjects(grid_x_-1, j, 0);
                                }
                            } else {
                                if (neighbors[kLEFT] != nullptr) {
                                    brothers = neighbors[kLEFT]->getPlacementObjects(grid_x_-1, j, k);
                                }
                            }
                        }
                    } else if (i == grid_x_) {
                        if (j < 0) {
                            if (k < 0) {
                                if (neighbors[kRIGHT_FLOOR_FRONT] != nullptr) {
                                    brothers = neighbors[kRIGHT_FLOOR_FRONT]->getPlacementObjects(0, grid_y_-1, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kRIGHT_FLOOR_BACK] != nullptr) {
                                    brothers = neighbors[kRIGHT_FLOOR_BACK]->getPlacementObjects(0, grid_y_-1, 0);
                                }
                            } else {
                                if (neighbors[kRIGHT_FLOOR] != nullptr) {
                                    brothers = neighbors[kRIGHT_FLOOR]->getPlacementObjects(0, grid_y_-1, k);
                                }
                            }
                        } else if (j == grid_y_) {
                            if (k < 0) {
                                if (neighbors[kRIGHT_ROOF_FRONT] != nullptr) {
                                    brothers = neighbors[kRIGHT_ROOF_FRONT]->getPlacementObjects(0, 0, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kRIGHT_ROOF_BACK] != nullptr) {
                                    brothers = neighbors[kRIGHT_ROOF_BACK]->getPlacementObjects(0, 0, 0);
                                }
                            } else {
                                if (neighbors[kRIGHT_ROOF] != nullptr) {
                                    brothers = neighbors[kRIGHT_ROOF]->getPlacementObjects(0, 0, k);
                                }
                            }
                        } else {
                            if (k < 0) {
                                if (neighbors[kRIGHT_FRONT] != nullptr) {
                                    brothers = neighbors[kRIGHT_FRONT]->getPlacementObjects(0, j, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kRIGHT_BACK] != nullptr) {
                                    brothers = neighbors[kRIGHT_BACK]->getPlacementObjects(0, j, 0);
                                }
                            } else {
                                if (neighbors[kRIGHT] != nullptr) {
                                    brothers = neighbors[kRIGHT]->getPlacementObjects(0, j, k);
                                }
                            }
                        }
                    } else if (j < 0) {
                        if (k < 0) {
                            if (neighbors[kFLOOR_FRONT] != nullptr) {
                                brothers = neighbors[kFLOOR_FRONT]->getPlacementObjects(i, grid_y_-1, grid_z_-1);
                            }
                        } else if (k == grid_z_) {
                            if (neighbors[kFLOOR_BACK] != nullptr) {
                                brothers = neighbors[kFLOOR_BACK]->getPlacementObjects(i, grid_y_-1, 0);
                            }
                        } else {
                            if (neighbors[kFLOOR] != nullptr) {
                                brothers = neighbors[kFLOOR]->getPlacementObjects(i, grid_y_-1, k);
                            }
                        }
                    } else if (j == grid_y_) {
                        if (k < 0) {
                            if (neighbors[kROOF_FRONT] != nullptr) {
                                brothers = neighbors[kROOF_FRONT]->getPlacementObjects(i, 0, grid_z_-1);
                            }
                        } else if (k == grid_z_) {
                            if (neighbors[kROOF_BACK] != nullptr) {
                                brothers = neighbors[kROOF_BACK]->getPlacementObjects(i, 0, 0);
                            }
                        } else {
                            if (neighbors[kROOF] != nullptr) {
                                brothers = neighbors[kROOF]->getPlacementObjects(i, 0, k);
                            }
                        }
                    } else if (k < 0) {
                        if (neighbors[kFRONT] != nullptr) {
                            brothers = neighbors[kFRONT]->getPlacementObjects(i, j, grid_z_-1);
                        }
                    } else if (k == grid_z_) {
                        if (neighbors[kBACK] != nullptr) {
                            brothers = neighbors[kBACK]->getPlacementObjects(i, j, 0);
                        }
                    } else {
                        brothers = grid_[i][j][k];
                    }

                    /* Merge targets with collisions array */
                    for (auto &obj : brothers) {
                        if (*obj != *target) {
                            tbb::mutex::scoped_lock lock_map(pivot_mutex);
                            grid_objects[obj->id()] = obj;
                        }
                    }

                }
            }
        }
    }

    /* Prepare vector for collision compute */
    std::vector<Model3D*> test_objects;
    for (auto &obj_pair : grid_objects) {
        test_objects.push_back(obj_pair.second);
    }

    /* Parallell collision loop for objects with cilkplus */
    std::map<int, std::vector<Model3D*>> recompute;
    cilk_for (auto i = 0; i < test_objects.size(); i++) {
        /* Abort program if object and room_object loop are same (must no happend) */
        assert(*target != *test_objects[i]);

        std::vector<Model3D*> collision_recompute = collision_engine_->DetectCollision(target, test_objects[i], object_mutex_);
        if (collision_recompute.size() > 0) {
            recompute[test_objects[i]->id()] = collision_recompute;
        }
    }

    /* end mutex lock */
    target->unlock();

    /* If collision detected, recompute older ones if exists */
    if (target->obstacle() != nullptr ) {
        try {
            for (auto &obj : recompute.at(target->obstacle()->id())) {
                using engine::core::ConfigEngine;
                if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_COLLISION) {
                    std::cout << "Recompute " << target->id() << "::" << obj->id() << std::endl;
                }
                cilk_spawn PivotCollision(obj, neighbors);
            }
        }
        /* Silently catched out-of-range (not a problem) */
        catch (const std::out_of_range& oor) {}
        
        cilk_sync;
    }
}

std::map<int, Model3D*> Model3D::GetNeighbors(Model3D* target, int level) noexcept
{
    std::map<int, Model3D*> neighbors;
    Model3D* tmp;

    auto i = target->placements()[0][0];
    auto j = target->placements()[0][1];
    auto k = target->placements()[0][2];

    if (i != 0) {
        if (j != 0) {
            if (k != 0 && grid_[i-1][j-1][k-1].size() != 0) {
                for (auto &n : grid_[i-1][j-1][k-1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (k != grid_z_-1 && grid_[i-1][j-1][k+1].size() != 0) {
                for (auto &n : grid_[i-1][j-1][k+1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (grid_[i-1][j-1][k].size() != 0) {
                for (auto &n : grid_[i-1][j-1][k]) {
                    neighbors[n->id()] = n;
                }
            }
        }

        if (j != grid_y_-1) {
            if (k != 0 && grid_[i-1][j+1][k-1].size() != 0) {
                for (auto &n : grid_[i-1][j+1][k-1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (k != grid_z_-1 && grid_[i-1][j+1][k+1].size() != 0) {
                for (auto &n : grid_[i-1][j+1][k+1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (grid_[i-1][j+1][k].size() != 0) {
                for (auto &n : grid_[i-1][j+1][k]) {
                    neighbors[n->id()] = n;
                }
            }
        }

        if (k != 0 && grid_[i-1][j][k-1].size() != 0) {
            for (auto &n : grid_[i-1][j][k-1]) {
                neighbors[n->id()] = n;
            }
        }

        if (k != grid_z_-1 && grid_[i-1][j][k+1].size() != 0) {
            for (auto &n : grid_[i-1][j][k+1]) {
                neighbors[n->id()] = n;
            }
        }

        if(grid_[i-1][j][k].size() != 0) {
            for (auto &n : grid_[i-1][j][k]) {
                neighbors[n->id()] = n;
            }
        }
    }

    if (i != grid_x_-1) {
        if (j != 0) {
            if (k != 0 && grid_[i+1][j-1][k-1].size() != 0) {
                for (auto &n : grid_[i+1][j-1][k-1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (k != grid_z_-1 && grid_[i+1][j-1][k+1].size() != 0) {
                for (auto &n : grid_[i+1][j-1][k+1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (grid_[i+1][j-1][k].size() != 0) {
                for (auto &n : grid_[i+1][j-1][k]) {
                    neighbors[n->id()] = n;
                }
            }
        }

        if (j != grid_y_-1) {
            if (k != 0 && grid_[i+1][j+1][k-1].size() != 0) {
                for (auto &n : grid_[i+1][j+1][k-1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (k != grid_z_-1 && grid_[i+1][j+1][k+1].size() != 0) {
                for (auto &n : grid_[i+1][j+1][k+1]) {
                    neighbors[n->id()] = n;
                }
            }

            if (grid_[i+1][j+1][k].size() != 0) {
                for (auto &n : grid_[i+1][j+1][k]) {
                    neighbors[n->id()] = n;
                }
            }
        }

        if (k != 0 && grid_[i+1][j][k-1].size() != 0) {
            for (auto &n : grid_[i+1][j][k-1]) {
                neighbors[n->id()] = n;
            }
        }

        if (k != grid_z_-1 && grid_[i+1][j][k+1].size() != 0) {
            for (auto &n : grid_[i+1][j][k+1]) {
                neighbors[n->id()] = n;
            }
        }

        if(grid_[i+1][j][k].size() != 0) {
            for (auto &n : grid_[i+1][j][k]) {
                neighbors[n->id()] = n;
            }
        }
    }

    if (j != 0) {
        if (k != 0 && grid_[i][j-1][k-1].size() != 0) {
            for (auto &n : grid_[i][j-1][k-1]) {
                neighbors[n->id()] = n;
            }
        }

        if (k != grid_z_-1 && grid_[i][j-1][k+1].size() != 0) {
            for (auto &n : grid_[i][j-1][k+1]) {
                neighbors[n->id()] = n;
            }
        }

        if (grid_[i][j-1][k].size() != 0) {
            for (auto &n : grid_[i][j-1][k]) {
                neighbors[n->id()] = n;
            }
        }
    }

    if (j != grid_y_-1) {
        if (k != 0 && grid_[i][j+1][k-1].size() != 0) {
            for (auto &n : grid_[i][j+1][k-1]) {
                neighbors[n->id()] = n;
            }
        }

        if (k != grid_z_-1 && grid_[i][j+1][k+1].size() != 0) {
            for (auto &n : grid_[i][j+1][k+1]) {
                neighbors[n->id()] = n;
            }
        }

        if (grid_[i][j+1][k].size() != 0) {
            for (auto &n : grid_[i][j+1][k]) {
                neighbors[n->id()] = n;
            }
        }
    }

    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        for (auto &n : grid_[i][j][k-1]) {
            neighbors[n->id()] = n;
        }
    }

    if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
        for (auto &n : grid_[i][j][k+1]) {
            neighbors[n->id()] = n;
        }
    }

    if (level) {
        std::map<int, Model3D*> neighbors_recur_merge;
        /* While level is not zero, merge neighbors maps */
        for (auto &n : neighbors) {
            auto neighbors_recur = GetNeighbors(n.second, level-1);
            neighbors_recur_merge.insert(neighbors_recur.begin(), neighbors_recur.end());
        }
        neighbors.insert(neighbors_recur_merge.begin(), neighbors_recur_merge.end());
    }

    return neighbors;
}

std::vector<Model3D*> Model3D::GetOrderNeighbors(Model3D* target) noexcept
{
    std::vector<Model3D*> neighbors(26, nullptr);
    auto i = target->placements()[0][0];
    auto j = target->placements()[0][1];
    auto k = target->placements()[0][2];

    if (i != 0) {
        if (j != 0) {
            if (k != 0 && grid_[i-1][j-1][k-1].size() != 0) {
                neighbors[kLEFT_FLOOR_FRONT] = grid_[i-1][j-1][k-1][0];
            }

            if (k != grid_z_-1 && grid_[i-1][j-1][k+1].size() != 0) {
                neighbors[kLEFT_FLOOR_BACK] = grid_[i-1][j-1][k+1][0];
            }

            if (grid_[i-1][j-1][k].size() != 0) {
                neighbors[kLEFT_FLOOR] = grid_[i-1][j-1][k][0];
            }
        }

        if (j != grid_y_-1) {
            if (k != 0 && grid_[i-1][j+1][k-1].size() != 0) {
                neighbors[kLEFT_ROOF_FRONT] = grid_[i-1][j+1][k-1][0];
            }

            if (k != grid_z_-1 && grid_[i-1][j+1][k+1].size() != 0) {
                neighbors[kLEFT_ROOF_BACK] = grid_[i-1][j+1][k+1][0];
            }

            if (grid_[i-1][j+1][k].size() != 0) {
                neighbors[kLEFT_ROOF] = grid_[i-1][j+1][k][0];
            }
        }

        if (k != 0 && grid_[i-1][j][k-1].size() != 0) {
            neighbors[kLEFT_FRONT] = grid_[i-1][j][k-1][0];
        }

        if (k != grid_z_-1 && grid_[i-1][j][k+1].size() != 0) {
            neighbors[kLEFT_BACK] = grid_[i-1][j][k+1][0];
        }

        if(grid_[i-1][j][k].size() != 0) {
            neighbors[kLEFT] = grid_[i-1][j][k][0];
        }
    }

    if (i != grid_x_-1) {
        if (j != 0) {
            if (k != 0 && grid_[i+1][j-1][k-1].size() != 0) {
                neighbors[kRIGHT_FLOOR_FRONT] = grid_[i+1][j-1][k-1][0];
            }

            if (k != grid_z_-1 && grid_[i+1][j-1][k+1].size() != 0) {
                neighbors[kRIGHT_FLOOR_BACK] = grid_[i+1][j-1][k+1][0];
            }

            if (grid_[i+1][j-1][k].size() != 0) {
                neighbors[kRIGHT_FLOOR] = grid_[i+1][j-1][k][0];
            }
        }

        if (j != grid_y_-1) {
            if (k != 0 && grid_[i+1][j+1][k-1].size() != 0) {
                neighbors[kRIGHT_ROOF_FRONT] = grid_[i+1][j+1][k-1][0];
            }

            if (k != grid_z_-1 && grid_[i+1][j+1][k+1].size() != 0) {
                neighbors[kRIGHT_ROOF_BACK] = grid_[i+1][j+1][k+1][0];
            }

            if (grid_[i+1][j+1][k].size() != 0) {
                neighbors[kRIGHT_ROOF] = grid_[i+1][j+1][k][0];
            }
        }

        if (k != 0 && grid_[i+1][j][k-1].size() != 0) {
            neighbors[kRIGHT_FRONT] = grid_[i+1][j][k-1][0];
        }

        if (k != grid_z_-1 && grid_[i+1][j][k+1].size() != 0) {
            neighbors[kRIGHT_BACK] = grid_[i+1][j][k+1][0];
        }

        if(grid_[i+1][j][k].size() != 0) {
            neighbors[kRIGHT] = grid_[i+1][j][k][0];
        }
    }

    if (j != 0) {
        if (k != 0 && grid_[i][j-1][k-1].size() != 0) {
            neighbors[kFLOOR_FRONT] = grid_[i][j-1][k-1][0];
        }

        if (k != grid_z_-1 && grid_[i][j-1][k+1].size() != 0) {
            neighbors[kFLOOR_BACK] = grid_[i][j-1][k+1][0];
        }

        if (grid_[i][j-1][k].size() != 0) {
            neighbors[kFLOOR] = grid_[i][j-1][k][0];
        }
    }

    if (j != grid_y_-1) {
        if (k != 0 && grid_[i][j+1][k-1].size() != 0) {
            neighbors[kROOF_FRONT] = grid_[i][j+1][k-1][0];
        }

        if (k != grid_z_-1 && grid_[i][j+1][k+1].size() != 0) {
            neighbors[kROOF_BACK] = grid_[i][j+1][k+1][0];
        }

        if (grid_[i][j+1][k].size() != 0) {
            neighbors[kROOF] = grid_[i][j+1][k][0];
        }
    }

    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        neighbors[kFRONT] = grid_[i][j][k-1][0];
    }

    if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
        neighbors[kBACK] = grid_[i][j][k+1][0];
    }

    return neighbors;
}

void Model3D::RecordHID()
{
    if (get_camera() != nullptr) {
        objects_[0]->RecordHID();
    }
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

} // namespace graphics

} // namespace engine
