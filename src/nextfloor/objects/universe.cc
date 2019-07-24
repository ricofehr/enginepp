/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Universe::Universe()
{
    using nextfloor::core::CommonServices;

    auto location = glm::vec3(0.0f);
    grid_ = CommonServices::getFactory()->MakeUniverseGrid(this);
    border_ = CommonServices::getFactory()->MakeBorder(location, grid_->scale());

    add_child(CommonServices::getFactory()->MakeRoom(location));
}

void Universe::Draw() noexcept
{
    assert(grid_ != nullptr);
    assert(border_ != nullptr);

    int active_index = 0;
    /* Lock current universe */
    lock();

    /* Select displayed rooms: 2 clipping levels or all rooms (clipping is 0) */
    display_rooms_.clear();

    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (cnt != active_index) {
            display_rooms_.push_back(objects_[cnt].get());
        }
    }

    /* Ensure active room is in first position */
    display_rooms_.insert(display_rooms_.begin(), objects_[active_index].get());

    /* Unlock mutex */
    unlock();

    /* Universe is only ready after 10 hops */
    if (ready()) {
        /* Draw Rooms on Gl Scene */
        for (auto &r : display_rooms_) {
            r->Draw();
        }
    }
}

} // namespace objects

} // namespace nextfloor
