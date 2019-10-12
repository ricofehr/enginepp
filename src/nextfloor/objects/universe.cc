/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe.h"

namespace nextfloor {

namespace objects {

Universe::Universe(std::unique_ptr<Grid> grid, std::unique_ptr<Border> border)
{
    grid_ = std::move(grid);
    border_ = std::move(border);
}

}  // namespace objects

}  // namespace nextfloor
