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
    using nextfloor::core::CommonServices;
    grid_ = CommonServices::getFactory()->MakeUniverseGrid(this);
    border_ = CommonServices::getFactory()->MakeBorder(glm::vec3(0.0f), grid_->scale());
}

void Universe::Draw() noexcept
{
    /* Universe is only ready after 10 hops */
    if (ready()) {
        DetectCollision();
        Move();
        PrepareDraw();

        /* Draw Rooms on Gl Scene */
        for (auto& r : objects_) {
            if (r->IsInCameraFieldOfView()) {
                r->Draw();
            }
        }
    }
}

}  // namespace objects

}  // namespace nextfloor
