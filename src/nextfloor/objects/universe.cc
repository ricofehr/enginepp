/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe.h"

namespace nextfloor {

namespace objects {

Universe::Universe(const MeshFactory& factory)
{
    grid_ = factory.MakeUniverseGrid(this);
    border_ = factory.MakeBorder(glm::vec3(0.0f), grid_->scale());
}

}  // namespace objects

}  // namespace nextfloor
