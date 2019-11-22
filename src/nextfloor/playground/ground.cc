/**
 *  @file ground.cc
 *  @brief Ground class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/ground.h"

#include <sstream>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace playground {

/* Move this function into Grid Object */
std::vector<nextfloor::mesh::Mesh*> Ground::FindCollisionNeighborsOf(const nextfloor::mesh::Mesh& target) const
{
    std::vector<nextfloor::mesh::Mesh*> all_neighbors(0);
    for (auto& coord : target.coords()) {
        auto neighbors = grid()->FindCollisionNeighbors(coord);
        all_neighbors.insert(all_neighbors.end(), neighbors.begin(), neighbors.end());
    }

    sort(all_neighbors.begin(), all_neighbors.end());
    all_neighbors.erase(unique(all_neighbors.begin(), all_neighbors.end()), all_neighbors.end());
    all_neighbors.erase(std::remove(all_neighbors.begin(), all_neighbors.end(), &target));

    tbb::mutex neighbors_mutex;
    std::vector<nextfloor::mesh::Mesh*> collision_neighbors(0);

    tbb::parallel_for(0, (int)all_neighbors.size(), 1, [&](int i) {
        auto neighbor = all_neighbors[i];
        if (target.IsNeighborEligibleForCollision(*neighbor)) {
            tbb::mutex::scoped_lock lock_map(neighbors_mutex);
            collision_neighbors.push_back(neighbor);
        }
    });

    return collision_neighbors;
}

nextfloor::mesh::Mesh* Ground::UpdateChildPlacement(nextfloor::mesh::Mesh* child)
{
    if (IsInside(*child)) {
        UpdateChildPlacementInGrid(child);
        return this;
    }
    else {
        return TransfertChildToNeighbor(child);
    }
}

void Ground::UpdateChildPlacementInGrid(nextfloor::mesh::Mesh* item)
{
    RemoveMeshToGrid(item);
    AddMeshToGrid(item);
}


nextfloor::mesh::Mesh* Ground::TransfertChildToNeighbor(nextfloor::mesh::Mesh* child)
{
    assert(parent_ != nullptr);
    assert(child != nullptr);

    return parent_->AddIntoChild(remove_child(child));
}

nextfloor::mesh::Mesh* Ground::AddIntoChild(std::unique_ptr<nextfloor::mesh::Mesh> mesh)
{
    assert(mesh != nullptr);
    tbb::mutex::scoped_lock lock_map(mutex_);

    auto mesh_raw = mesh.get();

    for (auto& object : objects_) {
        if (object->IsInside(*mesh_raw)) {
            object->add_child(std::move(mesh));
            return object.get();
        }
    }

    return nullptr;
}

bool Ground::IsInside(const nextfloor::mesh::Mesh& mesh) const
{
    return IsInside(mesh.location());
}

bool Ground::IsInside(const glm::vec3& location) const
{
    return grid()->IsInside(location);
}

nextfloor::mesh::Mesh* Ground::add_child(std::unique_ptr<nextfloor::mesh::Mesh> child)
{
    auto child_raw = CompositeMesh::add_child(std::move(child));
    AddMeshToGrid(child_raw);
    return child_raw;
}

std::unique_ptr<nextfloor::mesh::Mesh> Ground::remove_child(nextfloor::mesh::Mesh* child)
{
    RemoveMeshToGrid(child);
    return CompositeMesh::remove_child(child);
}

void Ground::RemoveMeshToGrid(nextfloor::mesh::Mesh* object)
{
    assert(grid_ != nullptr);
    if (object->hasChilds() && !object->hasLayout()) {
        for (auto& grant_child : object->childs()) {
            RemoveMeshToGrid(grant_child);
        }
    }
    else {
        grid_->RemoveMesh(object);
    }
}

void Ground::AddMeshToGrid(nextfloor::mesh::Mesh* child)
{
    assert(grid_ != nullptr);
    /* Place grid orphan grant child into ground grid */
    if (child->hasChilds() && !child->hasLayout()) {
        for (auto& grant_child : child->childs()) {
            // std::cout << "Grant child: " << grant_child->id() << std::endl;
            AddMeshToGrid(grant_child);
        }
    }
    else {
        grid_->AddItem(child);
    }
}

bool Ground::IsFrontPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsFrontPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool Ground::IsRightPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsRightPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool Ground::IsBackPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsBackPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool Ground::IsLeftPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsLeftPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool Ground::IsBottomPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsBottomPositionFilled()) {
            return true;
        }
    }

    return false;
}

bool Ground::IsTopPositionFilled() const
{
    for (auto& coord : coords_list_) {
        if (coord->IsTopPositionFilled()) {
            return true;
        }
    }

    return false;
}


// void CompositeMesh::InitChildsIntoGrid()
// {
//     for (auto& object : objects_) {
//         if (object->hasChilds()) {
//             object->InitChildsIntoGrid();
//         }
//         object->AddIntoAscendantGrid();
//     }
// }

// void CompositeMesh::AddIntoAscendantGrid()
// {
//     assert(parent_ != nullptr);
//     parent_->AddMeshToGrid(this);
// }


}  // namespace playground

}  // namespace nextfloor
