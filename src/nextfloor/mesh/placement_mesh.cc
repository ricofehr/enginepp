/**
 *  @file placement_mesh.cc
 *  @brief PlacementMesh class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/mesh/placement_mesh.h"

#include <tbb/tbb.h>
#include <glm/glm.hpp>
#include <vector>

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace mesh {

std::vector<glm::ivec3> PlacementMesh::coords() const
{
    std::vector<glm::ivec3> grid_coords(0);
    for (auto& box : coords_list_) {
        grid_coords.push_back(box->coords());
    }

    return grid_coords;
}

bool PlacementMesh::IsLastObstacle(Mesh* obstacle) const
{
    return obstacle_ == obstacle;
}

void PlacementMesh::UpdateObstacleIfNearer(Mesh* obstacle, float obstacle_distance)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    /* Update obstacle and distance if lower than former */
    if (obstacle_distance < move_factor()) {
        obstacle_ = obstacle;
        set_move_factor(obstacle_distance);
    }
}

void PlacementMesh::delete_gridcoord(GridBox* grid_box)
{
    for (int cnt = 0; cnt < coords_list_.size(); cnt++) {
        if (coords_list_[cnt] == grid_box) {
            coords_list_.erase(coords_list_.begin() + cnt);
            return;
        }
    }
}

std::vector<Mesh*> PlacementMesh::GetMovingObjects()
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

std::vector<glm::vec3> PlacementMesh::getCoordsModelMatrixComputed() const
{
    return border_->getCoordsModelMatrixComputed();
}

void PlacementMesh::ClearCoords()
{
    for (auto& box : coords_list_) {
        box->remove(this);
        delete_gridcoord(box);
    }
}

std::unique_ptr<Mesh> PlacementMesh::remove_child(Mesh* child)
{
    child->ClearCoords();
    return CompositeMesh::remove_child(child);
}

}  // namespace mesh

}  // namespace nextfloor
