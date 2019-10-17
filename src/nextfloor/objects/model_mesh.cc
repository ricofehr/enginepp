/**
 *  @file engine_model.cc
 *  @brief EngineModel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
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

void ModelMesh::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) { objects_[i]->PrepareDraw(view_projection_matrix); });

    tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) {
        polygons_[counter]->UpdateModelViewProjectionMatrix(view_projection_matrix);
    });
}

std::vector<std::pair<glm::mat4, std::string>> ModelMesh::GetModelViewProjectionsAndTextureToDraw() const
{
    std::vector<std::pair<glm::mat4, std::string>> mvps_with_texture;
    for (auto& polygon : polygons_) {
        mvps_with_texture.push_back(std::pair<glm::mat4, std::string>(polygon->mvp(), polygon->texture()));
    }
    return mvps_with_texture;
}

std::vector<Mesh*> ModelMesh::GetMovingObjects()
{
    std::vector<Mesh*> moving_objects;

    tbb::mutex movers_lock;

    if (IsMoved()) {
        moving_objects.push_back(this);
    }

    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) {
        auto movers = objects_[i]->GetMovingObjects();
        if (movers.size() != 0) {
            tbb::mutex::scoped_lock lock_map(movers_lock);
            moving_objects.insert(moving_objects.end(), movers.begin(), movers.end());
        }
    });

    return moving_objects;
}

std::vector<Mesh*> ModelMesh::FindCollisionNeighbors() const
{
    return parent_->FindCollisionNeighborsOf(*this);
}

/* Move this function into Grid Object */
std::vector<Mesh*> ModelMesh::FindCollisionNeighborsOf(const Mesh& target) const
{
    std::vector<Mesh*> all_neighbors(0);
    for (auto& coord : target.coords()) {
        auto neighbors = grid()->FindCollisionNeighbors(coord);
        all_neighbors.insert(all_neighbors.end(), neighbors.begin(), neighbors.end());
    }

    sort(all_neighbors.begin(), all_neighbors.end());
    all_neighbors.erase(unique(all_neighbors.begin(), all_neighbors.end()), all_neighbors.end());
    all_neighbors.erase(std::remove(all_neighbors.begin(), all_neighbors.end(), &target));

    tbb::mutex neighbors_mutex;
    std::vector<Mesh*> collision_neighbors(0);

    tbb::parallel_for(0, (int)all_neighbors.size(), 1, [&](int i) {
        auto neighbor = all_neighbors[i];
        if (target.IsNeighborEligibleForCollision(*neighbor)) {
            tbb::mutex::scoped_lock lock_map(neighbors_mutex);
            collision_neighbors.push_back(neighbor);
        }
    });

    return collision_neighbors;
}

bool ModelMesh::IsNeighborEligibleForCollision(const Mesh& neighbor) const
{
    return IsInDirection(neighbor) && IsNeighborReachable(neighbor);
}

bool ModelMesh::IsInDirection(const Mesh& target) const
{
    auto target_vector = target.location() - location();
    return glm::dot(movement(), target_vector) > 0;
}

bool ModelMesh::IsNeighborReachable(const Mesh& neighbor) const
{
    auto vector_neighbor = neighbor.location() - location();
    return glm::length(movement()) + glm::length(neighbor.movement())
           >= glm::length(vector_neighbor) - (diagonal() + neighbor.diagonal()) / 2.0f;
}

std::vector<Mesh*> ModelMesh::AllStubMeshs()
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

std::vector<glm::ivec3> ModelMesh::coords() const
{
    std::vector<glm::ivec3> grid_coords(0);
    for (auto& box : coords_list_) {
        grid_coords.push_back(box->coords());
    }

    return grid_coords;
}

void ModelMesh::MoveLocation()
{
    border_->ComputeNewLocation();

    tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) { polygons_[counter]->MoveLocation(); });
}

void ModelMesh::UpdateGridPlacement()
{
    if (parent_->IsInside(*this)) {
        parent_->UpdateChildPlacement(this);
    }
    else {
        parent_ = parent_->TransfertChildToNeighbor(this);
    }
}

bool ModelMesh::IsInside(const Mesh& mesh) const
{
    return IsInside(mesh.location());
}

bool ModelMesh::IsInside(const glm::vec3& location) const
{
    if (grid() == nullptr) {
        return false;
    }
    return grid()->IsInside(location);
}

void ModelMesh::UpdateChildPlacement(Mesh* item)
{
    RemoveItemsToGrid(item);
    AddMeshToGrid(item);
}

Mesh* ModelMesh::TransfertChildToNeighbor(Mesh* child)
{
    assert(parent_ != nullptr);
    assert(child != nullptr);

    return parent_->AddIntoChild(remove_child(child));
}

Mesh* ModelMesh::AddIntoChild(std::unique_ptr<Mesh> mesh)
{
    assert(mesh != nullptr);
    tbb::mutex::scoped_lock lock_map(mutex_);

    auto mesh_raw = mesh.get();

    for (auto& object : objects_) {
        if (object->IsInside(*mesh_raw)) {
            object->add_child(std::move(mesh));
            mesh_raw->AddIntoAscendantGrid();
            return object.get();
        }
    }

    return nullptr;
}

Mesh* ModelMesh::add_child(std::unique_ptr<Mesh> object)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    auto object_raw = object.get();
    object_raw->set_parent(this);

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

    return object_raw;
}

void ModelMesh::InitChildsIntoGrid()
{
    for (auto& object : objects_) {
        if (object->hasChilds()) {
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

void ModelMesh::AddMeshToGrid(Mesh* object)
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

std::unique_ptr<Mesh> ModelMesh::remove_child(Mesh* child)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    std::unique_ptr<Mesh> ret{nullptr};

    RemoveItemsToGrid(child);

    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (objects_[cnt].get() == child) {
            auto initial_count_childs = objects_.size();
            ret = std::move(objects_[cnt]);
            objects_.erase(objects_.begin() + cnt);
            /* Ensure child is erased from current objects_ array */
            assert(initial_count_childs == objects_.size() + 1);

            return ret;
        }
    }

    return ret;
}

void ModelMesh::RemoveItemsToGrid(Mesh* object)
{
    // assert(grid_ != nullptr);
    if (grid_ == nullptr) {
        parent_->RemoveItemsToGrid(object);
    }
    else {
        grid_->RemoveMesh(object);
    }
}


bool ModelMesh::IsLastObstacle(Mesh* obstacle) const
{
    return obstacle_ == obstacle;
}

void ModelMesh::UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    /* Update obstacle and distance if lower than former */
    if (obstacle_distance < border_->move_factor()) {
        obstacle_ = obstacle;
        border_->set_move_factor(-obstacle_distance);
        for (auto& polygon : polygons_) {
            polygon->set_move_factor(-obstacle_distance);
        }

        using nextfloor::core::CommonServices;
        if (CommonServices::getConfig()->IsCollisionDebugEnabled()) {
            LogCollision(*obstacle, obstacle_distance);
        }
    }
}

void ModelMesh::LogCollision(const Mesh& obstacle, float obstacle_distance)
{
    using nextfloor::core::CommonServices;

    std::ostringstream message;
    message << "Object::" << id() << " - Obstacle::" << obstacle.id();
    message << " - Distance::" << obstacle_distance;
    CommonServices::getLog()->WriteLine(std::move(message));
}

bool ModelMesh::IsFrontPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsFrontPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsRightPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsRightPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsBackPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsBackPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsLeftPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsLeftPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsBottomPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsBottomPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool ModelMesh::IsTopPositionFilled() const
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
