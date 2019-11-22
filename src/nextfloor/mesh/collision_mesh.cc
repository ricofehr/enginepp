/**
 *  @file collision_mesh.cc
 *  @brief CollisionMesh class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/mesh/collision_mesh.h"

#include <sstream>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace mesh {

std::vector<glm::ivec3> CollisionMesh::coords() const
{
    std::vector<glm::ivec3> grid_coords(0);
    for (auto& box : coords_list_) {
        grid_coords.push_back(box->coords());
    }

    return grid_coords;
}

bool CollisionMesh::IsLastObstacle(Mesh* obstacle) const
{
    return obstacle_ == obstacle;
}

void CollisionMesh::UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    /* Update obstacle and distance if lower than former */
    if (obstacle_distance < move_factor()) {
        obstacle_ = obstacle;
        set_move_factor(obstacle_distance);
        using nextfloor::core::CommonServices;
        if (CommonServices::getConfig()->IsCollisionDebugEnabled()) {
            LogCollision(*obstacle, obstacle_distance);
        }
    }
}

void CollisionMesh::LogCollision(const Mesh& obstacle, float obstacle_distance)
{
    using nextfloor::core::CommonServices;

    std::ostringstream message;
    message << "Object::" << id() << " - Obstacle::" << obstacle.id();
    message << " - Distance::" << obstacle_distance;
    CommonServices::getLog()->WriteLine(std::move(message));
}

void CollisionMesh::delete_gridcoord(GridBox* grid_box)
{
    for (int cnt = 0; cnt < coords_list_.size(); cnt++) {
        if (coords_list_[cnt] == grid_box) {
            coords_list_.erase(coords_list_.begin() + cnt);
            return;
        }
    }
}

std::vector<Mesh*> CollisionMesh::GetMovingObjects()
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

std::vector<glm::vec3> CollisionMesh::getCoordsModelMatrixComputed() const
{
    return border_->getCoordsModelMatrixComputed();
}

void CollisionMesh::ClearCoords()
{
    for (auto& box : coords_list_) {
        box->remove(this);
        delete_gridcoord(box);
    }
}

std::unique_ptr<nextfloor::mesh::Mesh> CollisionMesh::remove_child(nextfloor::mesh::Mesh* child)
{
    child->ClearCoords();
    return CompositeMesh::remove_child(child);
}

}  // namespace mesh

}  // namespace nextfloor
