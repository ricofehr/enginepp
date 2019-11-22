/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/playground/universe.h"

namespace nextfloor {

namespace playground {

Universe::Universe(std::unique_ptr<Grid> grid,
                   std::unique_ptr<nextfloor::mesh::Border> border,
                   std::vector<std::unique_ptr<Ground>> rooms)
{
    grid_ = std::move(grid);
    border_ = std::move(border);

    for (auto& room : rooms) {
        add_child(std::move(room));
    }
}

}  // namespace playground

}  // namespace nextfloor
