/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe.h"

#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Universe::Universe()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    grid_ = factory->MakeUniverseGrid(this);
    border_ = factory->MakeBorder(glm::vec3(0.0f), grid_->scale());
}

}  // namespace objects

}  // namespace nextfloor
