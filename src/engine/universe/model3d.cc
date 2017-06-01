/*
* Model3d class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cilk/cilk.h>

#include "engine/universe/camera.h"
#include "engine/physics/cl_collision_engine.h"
#include "engine/physics/serial_collision_engine.h"
#include "engine/physics/cilk_collision_engine.h"
#include "engine/core/config_engine.h"

namespace engine {
namespace universe {

//std::unique_ptr<engine::physics::CollisionEngine> Model3D::collision_engine_ = nullptr;

namespace {
    /* Unique id for object */
    static int sObjectId = 1;
}

/* Constructor */
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

/* Comparaison operator */
bool operator==(const Model3D& o1, const Model3D& o2)
{
    return o1.id() == o2.id();
}

bool operator!=(const Model3D& o1, const Model3D& o2)
{
    return o1.id() != o2.id();
}

/* Init CollisionEngine pointer */
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

/* Compute coords and move for the model */
void Model3D::PrepareDraw(Camera *cam)
{
    border_.set_distance(distance_);
    border_.MoveCoords();
    for (auto &element : elements_) {
        element->set_distance(distance_);
        element->ComputeMVP(cam);
    }
    distance_ = -1.0f;
    /* An object cant touch same object twice, except camera */
    id_last_collision_ = -1;
    if (!is_controlled_ && obstacle_ != nullptr) {
        id_last_collision_ = obstacle_->id();
    }
    obstacle_ = nullptr;
}

/* Draw the model */
void Model3D::Draw()
{
    /* Draw current object */
    for (auto &element : elements_) {
        element->Draw();
    }

    /* Draw associated objects ? */
//    for (auto &object : objects_) {
//        object->Draw();
//    }
}

/*
 * Allocate grid array dynamically
 */
void Model3D::InitGrid()
{
    grid_ = new std::vector<Model3D*> **[grid_y_];
    for (auto i = 0; i < grid_y_; i++) {
        grid_[i] = new std::vector<Model3D*> *[grid_x_];
        for (auto j = 0; j < grid_x_; j++) {
            grid_[i][j] = new std::vector<Model3D*>[grid_z_]();
        }
    }
}

/*
 *   Recompute the placement grid for objects into the room
 */
std::vector<std::unique_ptr<Model3D>> Model3D::ReinitGrid()
{
    std::vector<std::unique_ptr<Model3D>> ret;
    for (auto o = 0; o < objects_.size();) {
        if (!objects_[o]->IsMoved() && objects_[o]->get_placements().size() > 0) {
            ++o;
            continue;
        }
        /* check grid collision */
        engine::geometry::Box border = objects_[o]->border();
        std::vector<glm::vec3> coords = border.ComputeCoords();
        auto x1 = coords.at(0)[0];
        auto y1 = coords.at(0)[1];
        auto z1 = coords.at(0)[2];
        auto h1 = coords.at(3)[1] - y1;
        auto w1 = coords.at(1)[0] - x1;
        auto d1 = coords.at(4)[2] - z1;

        for (auto &p : objects_[o]->get_placements()) {
            tbb::mutex::scoped_lock lock(grid_mutex_);

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

        cilk_for (auto i = 0; i < grid_y_; i++) {
            auto y2 = grid_0[1] + (i+1) * grid_unit_y_;
            cilk_for (auto j = 0; j < grid_x_; j++) {
                auto x2 = grid_0[0] + j * grid_unit_x_;
                cilk_for (auto k = 0; k < grid_z_; k++) {
                    auto z2 = grid_0[2] + (k+1) * grid_unit_z_;
                    auto h2 = -grid_unit_y_;
                    auto w2 = grid_unit_x_;
                    auto d2 = -grid_unit_z_;

                    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
                        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                        tbb::mutex::scoped_lock lock(grid_mutex_);
                        objects_[o]->add_placement(i, j, k);
                        grid_[i][j][k].push_back(objects_[o].get());
                    }
                }
            }
        }

        if (objects_[o]->get_placements().size() == 0) {
            tbb::mutex::scoped_lock lock(object_mutex_);
            ret.push_back(std::move(objects_[o]));
            objects_.erase(objects_.begin() + o);
        } else {
            o++;
        }
    }

    return ret;
}

/* Display Placement Grid */
void Model3D::DisplayGrid()
{
    std::cout << "=== GRID FOR ID " << id_ << " ===" << std::endl << std::endl;
    for (auto i = 0; i < grid_y_; i++) {
        std::cout << "=== Floor " << i << std::endl;
        for (auto k = 0; k < grid_z_; k++) {
            for (auto j = 0; j < grid_x_; j++) {
                if (grid_[i][j][k].size() >0) {
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

/* Change object room */
std::unique_ptr<Model3D> Model3D::TransfertObject(std::unique_ptr<Model3D> obj, bool force) {
    if (force || IsInside(obj->location())) {
        /* Output if debug setted */
        using engine::core::ConfigEngine;
        if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_COLLISION && force) {
            std::cout << "Transfert " << obj->id() << " to Room " << id() << " (forced: " << force << ")" << std::endl;
        }
        /* Inverse object move if forced transfert */
        if (force) {
            obj->InverseMove();
        }
        /* Set current room active if its camera object */
        if (obj->type() == Model3D::kMODEL3D_CAMERA) {
            objects_.insert(objects_.begin(), std::move(obj));
        } else {
            objects_.push_back(std::move(obj));
        }

        return nullptr;
    }

    return obj;
}

/* Test if object is inside Room */
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

/* List outside objects into room (currently not used) */
std::vector<std::unique_ptr<Model3D>> Model3D::ListOutsideObjects()
{
    std::vector<std::unique_ptr<Model3D>> ret;
    for (auto &o : objects_) {
        if (!IsInside(o->location())) {
            ret.push_back(std::move(o));
        }
    }
    
    return ret;
}

/* Detect collisions inside current room */
void Model3D::DetectCollision(std::vector<Model3D*> neighbors)
{
    /* For all others moving objects
     Parallell loop with cilkplus */
    cilk_for (auto i = 0; i < objects_.size(); i++) {
        if (objects_[i]->IsMoved()) {
            PivotCollision(objects_[i].get(), neighbors);
        }
    }
}

/* Detect all collision for one Object */
void Model3D::PivotCollision(Model3D *object, std::vector<Model3D*> neighbors)
{
    tbb::mutex pivot_mutex;
    /* ensure that 2 same objects are not checked at the same time */
    object->lock();

    /* Prepare an unique objects collection for collision detection */
    std::map<int, Model3D*> grid_objects;

    /* Grid coordinates for current object */
    std::vector<std::vector<int>> placements = object->get_placements();

    /* Check all grid placements for current object and select other objects near this one */
    cilk_for (auto p = 0; p < placements.size(); p++) {
        auto y = placements[p][0];
        auto x = placements[p][1];
        auto z = placements[p][2];

        /* Other room objects adding */
        cilk_for (auto i = y-1; i <= y+1; i++) {
            cilk_for (auto j = x-1; j <= x+1; j++) {
                if ((i != -1 && i != grid_y_) || (j != -1 && j != grid_x_)) {
                    cilk_for (auto k = z-1; k <= z+1; k++) {
                        if (((i != -1 && i != grid_y_) || (k != -1 && k != grid_z_)) &&
                            ((j != -1 && j != grid_x_) || (k != -1 && k != grid_z_))) {

                            std::vector<Model3D*> targets;

                            if (i < 0) {
                                if (neighbors[kFLOOR] != nullptr) {
                                    targets = neighbors[kFLOOR]->getObjects(grid_y_-1, j, k);
                                }
                            } else if (i == grid_y_) {
                                if (neighbors[kROOF] != nullptr) {
                                    targets = neighbors[kROOF]->getObjects(0, j, k);
                                }
                            } else if (j < 0) {
                                if (neighbors[kLEFT] != nullptr) {
                                    targets = neighbors[kLEFT]->getObjects(i, grid_x_-1, k);
                                }
                            } else if (j == grid_x_) {
                                if (neighbors[kRIGHT] != nullptr) {
                                    targets = neighbors[kRIGHT]->getObjects(i, 0, k);
                                }
                            } else if (k < 0) {
                                if (neighbors[kFRONT] != nullptr) {
                                    targets = neighbors[kFRONT]->getObjects(i, j, grid_z_-1);
                                }
                            } else if (k == grid_z_) {
                                if (neighbors[kBACK] != nullptr) {
                                    targets = neighbors[kBACK]->getObjects(i, j, 0);
                                }
                            } else {
                                targets = grid_[i][j][k];
                            }

                            /* Merge targets with collisions array */
                            for (auto &obj : targets) {
                                if (*obj != *object) {
                                    tbb::mutex::scoped_lock lock_map(pivot_mutex);
                                    grid_objects[obj->id()] = obj;
                                }
                            }
                        } //fi: test on i and k, j and k
                    } //cilk_for k
                } //fi: test on i and j
            } //cilk_for j
        } //cilk_for i
    } //cilk_for placements

    /* Prepare vector for collision compute */
    std::vector<Model3D*> room_objects;
    for (auto & obj_pair : grid_objects) {
        room_objects.push_back(obj_pair.second);
    }

    /* Parallell collision loop for objects with cilkplus */
    std::map<int, std::vector<Model3D*>> recompute;
    cilk_for (auto i = 0; i < room_objects.size(); i++) {
        /* Abort program if object and room_object loop are same (must no happend) */
        assert(*object != *room_objects[i]);

        std::vector<Model3D*> collision_recompute = collision_engine_->DetectCollision(object, room_objects[i], object_mutex_);
        if (collision_recompute.size() > 0) {
            recompute[room_objects[i]->id()] = collision_recompute;
        }
    }

    /* end mutex lock */
    object->unlock();

    /* If collision detected, recompute older ones if exists */
    if (object->obstacle() != nullptr ) {
        try {
            for (auto & r : recompute.at(object->obstacle()->id())) {
                using engine::core::ConfigEngine;
                if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_COLLISION) {
                    std::cout << "Recompute " << object->id() << "::" << r->id() << std::endl;
                }
                cilk_spawn PivotCollision(r, neighbors);
            }
        }
        /* Silently catched out-of-range (not a problem) */
        catch (const std::out_of_range& oor) {}
        
        cilk_sync;
    }
}

/* Return a group of rooms near each other with a deeping level */
std::map<int, Model3D*> Model3D::GetNeighbors(Model3D *r, int level)
{
    std::map<int, Model3D*> neighbors;
    Model3D *tmp;

    auto i = r->get_placements()[0][0];
    auto j = r->get_placements()[0][1];
    auto k = r->get_placements()[0][2];

    if (i != 0 && grid_[i-1][j][k].size() != 0) {
        tmp = grid_[i-1][j][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (i != grid_y_-1 && grid_[i+1][j][k].size() != 0) {
        tmp = grid_[i+1][j][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (j != 0 && grid_[i][j-1][k].size() != 0) {
        tmp = grid_[i][j-1][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (j != grid_x_-1 && grid_[i][j+1][k].size() != 0) {
        tmp = grid_[i][j+1][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        tmp = grid_[i][j][k-1][0];
        neighbors[tmp->id()] = tmp;
    }

    if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
        tmp = grid_[i][j][k+1][0];
        neighbors[tmp->id()] = tmp;
    }

    if (level) {
        std::map<int, Model3D*> neighbors_recur_merge;
        for (auto &n : neighbors) {
            auto neighbors_recur = GetNeighbors(n.second, level-1);
            neighbors_recur_merge.insert(neighbors_recur.begin(), neighbors_recur.end());
        }
        neighbors.insert(neighbors_recur_merge.begin(), neighbors_recur_merge.end());
    }

    return neighbors;
}

/* Return the 6 rooms (if all exists) near the targetting room (order by common side) */
std::vector<Model3D*> Model3D::GetOrderNeighbors(Model3D *r)
{
    std::vector<Model3D*> neighbors(6, nullptr);
    auto i = r->get_placements()[0][0];
    auto j = r->get_placements()[0][1];
    auto k = r->get_placements()[0][2];

    if (i != 0 && grid_[i-1][j][k].size() != 0) {
        neighbors[kFLOOR] = grid_[i-1][j][k][0];
    }

    if (i != grid_y_-1 && grid_[i+1][j][k].size() != 0) {
        neighbors[kROOF] = grid_[i+1][j][k][0];
    }

    if (j != 0 && grid_[i][j-1][k].size() != 0) {
        neighbors[kLEFT] = grid_[i][j-1][k][0];
    }

    if (j != grid_x_-1 && grid_[i][j+1][k].size() != 0) {
        neighbors[kRIGHT] = grid_[i][j+1][k][0];
    }
    
    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        neighbors[kFRONT] = grid_[i][j][k-1][0];
    }
    
    if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
        neighbors[kBACK] = grid_[i][j][k+1][0];
    }
    
    return neighbors;
}

/* Move camera associated to current object */
void Model3D::MoveCamera()
{
    if (get_camera() != nullptr) {
        dynamic_cast<Camera*>(objects_[0].get())->Move();
    }
}

/* Destructor, empty grid_ array */
Model3D::~Model3D()
{
    if (grid_ != nullptr) {
        for (auto i = 0; i < grid_y_; i++) {
            for (auto j = 0; j < grid_x_; j++) {
                delete [] grid_[i][j];
            }
            delete [] grid_[i];
        }
        delete [] grid_;
    }
}

}//namespace geometry
}//namespace engine
