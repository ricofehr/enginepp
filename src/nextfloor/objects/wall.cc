#include "nextfloor/objects/wall.h"

#include <memory>
#include <iostream>

namespace nextfloor {

namespace objects {

void Wall::AddBricks(std::vector<std::unique_ptr<Mesh>> wall_bricks)
{
    for (auto& wall_brick : wall_bricks) {
        add_child(std::move(wall_brick));
    }
}

}  // namespace objects

}  // namespace nextfloor
