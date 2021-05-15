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


std::vector<Mesh*> PlacementMesh::GetMovingObjects()
{
    std::vector<Mesh*> moving_objects;

    tbb::mutex movers_lock;

    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) {
        auto movers = objects_[i]->GetMovingObjects();
        if (movers.size() != 0) {
            tbb::mutex::scoped_lock lock_map(movers_lock);
            moving_objects.insert(moving_objects.end(), movers.begin(), movers.end());
        }
    });
    return moving_objects;
}

std::unique_ptr<Mesh> PlacementMesh::remove_child(Mesh* child)
{
    child->ClearCoords();
    return CompositeMesh::remove_child(child);
}

}  // namespace mesh

}  // namespace nextfloor
