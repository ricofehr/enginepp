#include "nextfloor/playground/wall.h"

#include <memory>
#include <iostream>

namespace nextfloor {

namespace playground {

void Wall::AddBricks(std::vector<std::unique_ptr<nextfloor::thing::Thing>> wall_bricks)
{
    for (auto& wall_brick : wall_bricks) {
        add_child(std::move(wall_brick));
    }
}

std::vector<nextfloor::mesh::Mesh*> Wall::FindCollisionNeighborsOf(const Mesh& target) const
{
    assert(parent_ != nullptr);
    return parent_->FindCollisionNeighborsOf(target);
}

std::unique_ptr<nextfloor::mesh::Mesh> Wall::remove_child(nextfloor::mesh::Mesh* child)
{
    child->ClearCoords();
    return CompositeMesh::remove_child(child);
}


}  // namespace playground

}  // namespace nextfloor
