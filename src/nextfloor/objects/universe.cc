/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/objects/universe.h"

#include <GLFW/glfw3.h>
#include <map>
#include <iostream>
#include <tbb/tbb.h>

#include "nextfloor/objects/engine_border.h"
#include "nextfloor/objects/engine_grid.h"
#include "nextfloor/objects/room.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace objects {

Universe::Universe()
{
    type_ = kMODEL_UNIVERSE;
}

void Universe::InitGrid(std::unique_ptr<EngineGrid> grid) noexcept
{
    grid_ = std::move(grid);
}

void Universe::Draw() noexcept
{
    assert(grid_ != nullptr);
    assert(border_ != nullptr);
    //using nextfloor::core::CommonServices;

    //auto clipping = CommonServices::getConfig()->getClippingLevel();

    int active_index = 0;
    /* Detect current room */
    // int active_index = -1;
    // tbb::parallel_for (0, (int)objects_.size(), 1, [&](int cnt) {
    //     if (objects_[cnt]->get_camera() != nullptr) {
    //         active_index = cnt;
    //     }
    // });

    // /* if no active Room, return */
    // if (active_index == -1) {
    //     return;
    // }

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

void Universe::InitDoorsForRooms() noexcept
{
    grid_->InitDoorsAndWindows();
}

} // namespace objects

} // namespace nextfloor
