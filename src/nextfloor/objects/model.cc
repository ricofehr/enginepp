/**
 *  @file engine_model.cc
 *  @brief EngineModel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *
 */

#include "nextfloor/objects/model.h"

#include <SOIL/SOIL.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"
#include "nextfloor/renderer/game_window.h"

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
    /* Draw meshes of current object */
    for (auto &mesh : meshes_) {
        mesh->UpdateModelViewProjectionMatrix();
        mesh->Draw();
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
    grid_->AddItemToGrid(object_raw);
    /* Ensure object is well added */
    assert(objects_.size() == initial_objects_size + 1);
    unlock();

    return object_raw;
}

std::unique_ptr<EngineObject> Model::remove_child(EngineObject* child) noexcept
{
    std::unique_ptr<EngineObject> ret{nullptr};

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

} // namespace objects

} // namespace nextfloor
