/**
 *  @file dynamic_mesh.cc
 *  @brief DynamicMesh class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/mesh/dynamic_mesh.h"

#include <glm/glm.hpp>
#include <tbb/tbb.h>

namespace nextfloor {

namespace mesh {

std::vector<Mesh*> DynamicMesh::FindCollisionNeighbors() const
{
    assert(parent_ != nullptr);
    return parent_->FindCollisionNeighborsOf(*this);
}

bool DynamicMesh::IsNeighborEligibleForCollision(const Mesh& neighbor) const
{
    return IsInDirection(neighbor) && IsNeighborReachable(neighbor);
}

bool DynamicMesh::IsInDirection(const Mesh& target) const
{
    glm::vec3 target_vector = target.location() - location();
    return glm::dot(movement(), target_vector) > 0;
}

bool DynamicMesh::IsNeighborReachable(const Mesh& neighbor) const
{
    glm::vec3 vector_neighbor = neighbor.location() - location();
    return glm::length(movement()) + glm::length(neighbor.movement())
           >= glm::length(vector_neighbor) - (diagonal() + neighbor.diagonal()) / 2.0f;
}

void DynamicMesh::MoveLocation()
{
    border_->ComputeNewLocation();
    tbb::parallel_for(0, static_cast<int>(polygons_.size()), 1, [&](int counter) { polygons_[counter]->MoveLocation(); });
    parent_ = parent_->UpdateChildPlacement(this);
}

float DynamicMesh::move_factor() const
{
    return border_->move_factor();
}

void DynamicMesh::set_movement(const glm::vec3& movement)
{
    border_->set_movement(movement);
    for (auto& object : objects_) {
        object->set_movement(movement);
    }

    for (auto& polygon : polygons_) {
        polygon->set_movement(movement);
    }
}

void DynamicMesh::set_move_factor(float move_factor)
{
    border_->set_move_factor(-move_factor);

    for (auto& object : objects_) {
        object->set_move_factor(move_factor);
    }

    for (auto& polygon : polygons_) {
        polygon->set_move_factor(-move_factor);
    }
}

}  // namespace mesh

}  // namespace nextfloor
