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

void Universe::Draw()
{
    /* Universe is only ready after 10 hops */
    if (ready()) {
        DetectCollision();
        Move();
        PrepareDraw();

        /* Draw Rooms on Gl Scene */
        assert(active_camera_ != nullptr);
        for (auto& r : objects_) {
            if (active_camera_->IsInFieldOfView(*r)) {
                r->Draw();
            }
        }
    }
}

}  // namespace objects

}  // namespace nextfloor
