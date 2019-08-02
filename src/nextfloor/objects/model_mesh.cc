/**
 *  @file engine_model.cc
 *  @brief EngineModel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *
 */

#include "nextfloor/objects/model_mesh.h"

#include <sstream>

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace objects {

namespace {
    /* Unique id for object */
    static int sObjectId = 1;
} // anonymous namespace

ModelMesh::ModelMesh()
{
    id_ = sObjectId++;
}

bool operator==(const ModelMesh& o1, const ModelMesh& o2)
{
    return o1.id_ == o2.id_;
}

bool operator!=(const ModelMesh& o1, const ModelMesh& o2)
{
    return o1.id_ != o2.id_;
}

void ModelMesh::Draw() noexcept
{
    PrepareDraw();

    /* Draw meshes of current object */
    for (auto &mesh : polygons_) {
        mesh->UpdateModelViewProjectionMatrix();
        mesh->Draw(renderer_);
    }

    /* Draw childs objects */
    for (auto &object : objects_) {
        object->Draw();
    }
}

Mesh* ModelMesh::AddIntoChild(std::unique_ptr<Mesh> mesh) noexcept
{
    for (auto &object : objects_) {
        if (object->IsInside(mesh.get())) {
            return add_child(std::move(mesh));
        }
    }

    return nullptr;
}

bool ModelMesh::IsInside(Mesh* mesh) noexcept
{
    return grid()->IsInside(mesh->location());
}

Mesh* ModelMesh::add_child(std::unique_ptr<Mesh> object) noexcept
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

void ModelMesh::AddItemToGrid(Mesh* object) noexcept
{
    if (grid_ == nullptr) {
        if (parent_ != nullptr) {
            dynamic_cast<ModelMesh*>(parent_)->AddItemToGrid(object);
        }
    } else {
        auto coords_list = grid_->AddItemToGrid(object);
        dynamic_cast<ModelMesh*>(object)->set_gridcoords(coords_list);
    }
}

std::unique_ptr<Mesh> ModelMesh::remove_child(Mesh* child) noexcept
{
    std::unique_ptr<Mesh> ret{nullptr};

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

void ModelMesh::RemoveItemToGrid(Mesh* object) noexcept
{
    if (grid_ == nullptr) {
        if (parent_ != nullptr) {
            dynamic_cast<ModelMesh*>(parent_)->RemoveItemToGrid(object);
        }
    } else {
        grid_->RemoveItemToGrid(object);
    }
}

bool ModelMesh::IsLastObstacle(Mesh* obstacle) const noexcept
{
    return obstacle_ == obstacle;
}

void ModelMesh::UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance) noexcept
{
    /* Update obstacle and distance if lower than former */
    lock();
    if (obstacle_distance < border_->distance()) {
        obstacle_ = obstacle;
        border_->set_distance(-obstacle_distance);

        using nextfloor::core::CommonServices;
        if (CommonServices::getConfig()->IsCollisionDebugEnabled()) {
            LogCollision(obstacle, obstacle_distance);
        }
    }
    unlock();
}

void ModelMesh::LogCollision(Mesh* obstacle, float obstacle_distance)
{
    using nextfloor::core::CommonServices;

    std::ostringstream message;
    message << "Object::" << id() << " - Obstacle::" << obstacle->id();
    message << " - Distance::" << obstacle_distance;
    CommonServices::getLog()->WriteLine(std::move(message));
}

void ModelMesh::TransferCameraToOtherMesh(Mesh* other)
{
    other->set_camera(std::move(camera_));
    camera_ = nullptr;
}

Camera* ModelMesh::camera() const noexcept
{
    if (camera_ == nullptr) {
        return nullptr;
    }
    return camera_.get();
}

} // namespace objects

} // namespace nextfloor
