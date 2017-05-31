/*
* Model3d class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

#include <iostream>
#include <fstream>
#include <vector>
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
            if (objects_[o]->type() == Model3D::kMODEL3D_CAMERA) {
                cam_ = nullptr;
            }
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
            cam_ = (Camera *)obj.get();
        }

        objects_.push_back(std::move(obj));
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

/* Move camera associated to current object */
void Model3D::MoveCamera()
{
    if (cam_ != nullptr) {
        cam_->Move();
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
