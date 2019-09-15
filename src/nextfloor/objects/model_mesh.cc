/**
 *  @file engine_model.cc
 *  @brief EngineModel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 *
 *
 */

#include "nextfloor/objects/model_mesh.h"

#include <sstream>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace objects {

namespace {
/* Unique id for object */
static int sObjectId = 1;
}  // anonymous namespace

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
    /* Custom overrided actions before drawing object */
    PrepareDraw();

    /* Draw meshes of current object */
    for (auto& polygon : polygons_) {
        polygon->Draw(renderer_);
    }

    for (auto& object : objects_) {
        object->Draw();
    }
}

void ModelMesh::DetectCollision() noexcept
{
    if (IsMoved()) {
        PivotCollision();
    }

    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) { objects_[i]->DetectCollision(); });
}

void ModelMesh::PivotCollision() noexcept
{
    using nextfloor::core::CommonServices;
    static CollisionEngine* collision_engine = CommonServices::getFactory()->MakeCollisionEngine();

    /* Prepare vector for collision compute */
    std::vector<Mesh*> test_objects = parent_->FindCollisionNeighborsOf(this);

    /* Parallell collision loop for objects with tbb */
    tbb::parallel_for(0, (int)test_objects.size(), 1, [&](int i) {
        assert(*this != *(dynamic_cast<ModelMesh*>(test_objects[i])));
        collision_engine->DetectCollision(this, test_objects[i]);
    });
}

std::vector<Mesh*> ModelMesh::FindCollisionNeighborsOf(Mesh* target) const noexcept
{
    std::vector<Mesh*> all_neighbors(0);
    for (auto& coord : target->coords()) {
        auto neighbors = grid()->FindCollisionNeighbors(coord);
        all_neighbors.insert(all_neighbors.end(), neighbors.begin(), neighbors.end());
    }

    sort(all_neighbors.begin(), all_neighbors.end());
    all_neighbors.erase(unique(all_neighbors.begin(), all_neighbors.end()), all_neighbors.end());
    all_neighbors.erase(std::remove(all_neighbors.begin(), all_neighbors.end(), target));

    std::vector<Mesh*> collision_neighbors(0);
    for (auto& neighbor : all_neighbors) {
        if (target->IsNeighborEligibleForCollision(neighbor)) {
            collision_neighbors.push_back(neighbor);
        }
    }

    return collision_neighbors;
}

bool ModelMesh::IsNeighborEligibleForCollision(Mesh* neighbor) const
{
    return IsInDirection(neighbor) && IsNeighborReachable(neighbor);
}

bool ModelMesh::IsNeighborReachable(Mesh* neighbor) const
{
    auto vector_neighbor = neighbor->location() - location();
    return glm::length(movement()) + glm::length(neighbor->movement())
           >= glm::length(vector_neighbor) - (diagonal() + neighbor->diagonal()) / 2.0f;
}

bool ModelMesh::IsInDirection(Mesh* target) const
{
    auto target_vector = target->location() - location();
    return glm::dot(movement(), target_vector) > 0;
}

std::vector<Mesh*> ModelMesh::AllStubMeshs() noexcept
{
    std::vector<Mesh*> meshes(0);

    if (objects_.size() == 0) {
        meshes.push_back(this);
    }
    else {
        for (auto& object : objects_) {
            auto object_meshs = object->AllStubMeshs();
            meshes.insert(meshes.end(), object_meshs.begin(), object_meshs.end());
        }
    }

    return meshes;
}

std::vector<glm::ivec3> ModelMesh::coords()
{
    std::vector<glm::ivec3> grid_coords(0);
    for (auto& box : coords_list_) {
        grid_coords.push_back(box->coords());
    }

    return grid_coords;
}

void ModelMesh::Move() noexcept
{
    if (IsMoved()) {
        border_->ComputeNewLocation();

        tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) {
            polygons_[counter]->MoveLocation();
        });
    }

    tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) {
        polygons_[counter]->UpdateModelViewProjectionMatrix();
    });

    /*
     *  Compute GL coords for childs
     *  TransfertChild (called by Move) can move child to another parent
     *  So, need to have a sequential loop
     */
    for (auto counter = 0; counter < objects_.size();) {
        auto nb_objs = objects_.size();

        objects_[counter]->Move();

        if (nb_objs == objects_.size()) {
            counter++;
        }
    }

    if (IsMoved()) {
        if (parent_->IsInside(this)) {
            parent_->UpdateChildPlacement(this);
        }
        else {
            parent_ = parent_->TransfertChildToNeighbor(this);
        }
    }
}

bool ModelMesh::IsInside(Mesh* mesh) noexcept
{
    assert(grid() != nullptr);
    return grid()->IsInside(mesh->location());
}

void ModelMesh::UpdateChildPlacement(Mesh* item) noexcept
{
    RemoveItemsToGrid(item);
    AddMeshToGrid(item);
}

Mesh* ModelMesh::TransfertChildToNeighbor(Mesh* child) noexcept
{
    assert(parent_ != nullptr);
    assert(child != nullptr);

    return parent_->AddIntoChild(remove_child(child));
}

Mesh* ModelMesh::AddIntoChild(std::unique_ptr<Mesh> mesh) noexcept
{
    assert(mesh != nullptr);

    auto mesh_raw = mesh.get();

    for (auto& object : objects_) {
        if (object->IsInside(mesh_raw)) {
            object->add_child(std::move(mesh));
            mesh_raw->AddIntoAscendantGrid();
            return object.get();
        }
    }

    return nullptr;
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
    }
    else {
        objects_.push_back(std::move(object));
    }

    /* Ensure object is well added */
    assert(objects_.size() == initial_objects_size + 1);
    unlock();

    return object_raw;
}

void ModelMesh::InitChildsIntoGrid()
{
    for (auto& object : objects_) {
        if (!object->hasNoChilds()) {
            object->InitChildsIntoGrid();
        }
        object->AddIntoAscendantGrid();
    }
}

void ModelMesh::AddIntoAscendantGrid()
{
    assert(parent_ != nullptr);
    parent_->AddMeshToGrid(this);
}

void ModelMesh::AddMeshToGrid(Mesh* object) noexcept
{
    if (grid_ == nullptr) {
        assert(parent_ != nullptr);
        parent_->AddMeshToGrid(object);
    }
    else {
        auto coords_list = grid_->AddItem(object);
        dynamic_cast<ModelMesh*>(object)->set_gridcoords(coords_list);
    }
}

std::unique_ptr<Mesh> ModelMesh::remove_child(Mesh* child) noexcept
{
    std::unique_ptr<Mesh> ret{nullptr};

    RemoveItemsToGrid(child);

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

void ModelMesh::RemoveItemsToGrid(Mesh* object) noexcept
{
    // assert(grid_ != nullptr);
    if (grid_ == nullptr) {
        parent_->RemoveItemsToGrid(object);
    }
    else {
        grid_->RemoveMesh(object);
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
    if (obstacle_distance < border_->move_factor()) {
        obstacle_ = obstacle;
        border_->set_move_factor(-obstacle_distance);
        for (auto& polygon : polygons_) {
            polygon->set_move_factor(-obstacle_distance);
        }

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

bool ModelMesh::IsFrontPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsFrontPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsRightPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsRightPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsBackPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsBackPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsLeftPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsLeftPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsBottomPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsBottomPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsTopPositionFilled() const noexcept
{
    for (auto& coord : coords_list_) {
        if (coord->IsTopPositionFilled()) {
            return true;
        }
    }

    return false;
}

}  // namespace objects

}  // namespace nextfloor
