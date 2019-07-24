/**
 *  @file engine_model.cc
 *  @brief EngineModel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *
 */

#include "nextfloor/objects/model.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace objects {

namespace {
    /* Unique id for object */
    static int sObjectId = 1;
} // anonymous namespace

Model::Model()
{
    id_ = sObjectId++;
}

bool operator==(const Model& o1, const Model& o2)
{
    return o1.id_ == o2.id_;
}

bool operator!=(const Model& o1, const Model& o2)
{
    return o1.id_ != o2.id_;
}

void Model::Draw() noexcept
{
    PrepareDraw();

    /* Draw meshes of current object */
    for (auto &mesh : meshes_) {
        mesh->UpdateModelViewProjectionMatrix();
        mesh->Draw(renderer_);
    }

    /* Draw childs objects */
    for (auto &object : objects_) {
        object->Draw();
    }
}

EngineObject* Model::add_child(std::unique_ptr<EngineObject> object) noexcept
{
    auto object_raw = object.get();
    object_raw->set_parent(this);

    lock();
    auto initial_objects_size = objects_.size();
    /* Insert Camera as first element. Push on the stack for others */
    if (object_raw->IsCamera()) {
        objects_.insert(objects_.begin(), std::move(object));
    } else {
        objects_.push_back(std::move(object));
    }

    AddItemToGrid(object_raw);
    /* Ensure object is well added */
    assert(objects_.size() == initial_objects_size + 1);
    unlock();

    return object_raw;
}

void Model::AddItemToGrid(EngineObject* object) noexcept
{
    if (grid_ == nullptr) {
        if (parent_ != nullptr) {
            dynamic_cast<Model*>(parent_)->AddItemToGrid(object);
        }
    } else {
        auto coords_list = grid_->AddItemToGrid(object);
        dynamic_cast<Model*>(object)->set_gridcoords(coords_list);
    }
}

std::unique_ptr<EngineObject> Model::remove_child(EngineObject* child) noexcept
{
    std::unique_ptr<EngineObject> ret{nullptr};

    RemoveItemToGrid(child);
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

void Model::RemoveItemToGrid(EngineObject* object) noexcept
{
    if (grid_ == nullptr) {
        if (parent_ != nullptr) {
            dynamic_cast<Model*>(parent_)->RemoveItemToGrid(object);
        }
    } else {
        grid_->RemoveItemToGrid(object);
    }
}

bool Model::IsLastObstacle(EngineObject* obstacle) const noexcept
{
    return obstacle_ == obstacle;
}

void Model::UpdateObstacleIfNearer(EngineObject* obstacle, float obstacle_distance) noexcept
{
    /* Update obstacle and distance if lower than former */
    lock();
    if (obstacle_distance < border_->distance()) {
        obstacle_ = obstacle;
        border_->set_distance(-obstacle_distance);

        /* Print debug if setting */
        using nextfloor::core::CommonServices;
        if (CommonServices::getConfig()->getDebugLevel() >= CommonServices::getLog()->kDEBUG_COLLISION) {
            std::cerr << "Object::" << id() << " - Obstacle::" << obstacle->id() << " - Distance::" << obstacle_distance << std::endl;
        }
    }
    unlock();
}

} // namespace objects

} // namespace nextfloor
