/**
 *  @file universe.cc
 *  @brief Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/universe/universe.h"

#include <GLFW/glfw3.h>
#include <map>
#include <iostream>
#include <tbb/tbb.h>

#include "nextfloor/graphics/border.h"
#include "nextfloor/universe/room.h"
#include "nextfloor/universe/factory/random_universe_factory.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace universe {

namespace {
    /* Used for delay between 2 objects creation */
    static double sLastTime = 0;
} // anonymous namespace

Universe::Universe()
{
    type_ = kMODEL3D_UNIVERSE;

    /* Init Grid Settings */
    grid_x_ = kGRID_X;
    grid_y_ = kGRID_Y;
    grid_z_ = kGRID_Z;
    grid_unit_x_ = kGRID_UNIT_X;
    grid_unit_y_ = kGRID_UNIT_Y;
    grid_unit_z_ = kGRID_UNIT_Z;
    InitGrid();

    using nextfloor::graphics::Border;
    border_ = std::make_unique<Border>(glm::vec3(grid_x_*grid_unit_x_/2,
                                                 grid_y_*grid_unit_y_/2,
                                                 grid_z_*grid_unit_z_/2),
                                       glm::vec4(0,0,0,0));
}

void Universe::Draw() noexcept
{
    using nextfloor::core::CommonServices;

    auto clipping = CommonServices::getConfig().getSetting<int>("clipping");

    /* Detect current room */
    int active_index = -1;
    tbb::parallel_for (0, (int)objects_.size(), 1, [&](int cnt) {
        if (objects_[cnt]->get_camera() != nullptr) {
            active_index = cnt;
        }
    });

    /* if no active Room, return */
    if (active_index == -1) {
        return;
    }

    /* Record moving orders for camera */
    //objects_[active_index]->RecordHID();

    /* Lock current universe */
    lock();

    /* Select displayed rooms: 2 clipping levels or all rooms (clipping is 0) */
    display_rooms_.clear();
    if (clipping > 0) {
        display_rooms_ = objects_[active_index]->FindClippingNeighbors(clipping);
    } else {
        for (auto cnt = 0; cnt < objects_.size(); cnt++) {
            if (cnt != active_index) {
                display_rooms_.push_back(objects_[cnt].get());
            }
        }
    }
    /* Ensure active room is in first position */
    display_rooms_.insert(display_rooms_.begin(), objects_[active_index].get());

    /* Unlock mutex */
    unlock();

    /* Universe is only ready after 10 hops */
    if (ready()) {
        /* Detect collision in display rooms */
        tbb::parallel_for (0, (int)display_rooms_.size(), 1, [&](int cnt) {
            display_rooms_[cnt]->DetectCollision();
        });

        /* Compute new coords after current move, first the active room */
        display_rooms_[0]->Move();
        tbb::parallel_for (1, (int)display_rooms_.size(), 1, [&](int cnt) {
            display_rooms_[cnt]->Move();
        });

        /* Draw Rooms on Gl Scene */
        for (auto &r : display_rooms_) {
            r->Draw();
        }
    }

    /* GL functions during object generate, then needs serial execution */
    float freq = 0.0f;
    if ((freq = CommonServices::getConfig().getSetting<float>("load_objects_freq")) > 0.0f) {
        nextfloor::universe::factory::RandomUniverseFactory factory;
        double current_time = glfwGetTime();
        for (auto &r : objects_) {
            if (!r->IsFull() && current_time - sLastTime >= freq) {
                /* Generate a new Brick in Room r */
                auto room = dynamic_cast<Room*>(r.get());
                factory.GenerateBrick(room);
            }
        }

        if (current_time - sLastTime >= freq) {
            sLastTime = current_time;
        }
    }
}

void Universe::InitDoorsForRooms() noexcept
{
    for (auto &o : objects_) {
        /* Objects contained in Universe are Rooms, with Doors and Windows */
        auto r = dynamic_cast<Room*>(o.get());

        auto i = r->placements()[0][0];
        auto j = r->placements()[0][1];
        auto k = r->placements()[0][2];

        if (i != 0 && IsPositionInTheGridEmpty(i-1,j,k) == kGRID_USED) {
            r->addDoor(kLEFT);
        }

        if (i != grid_x_-1 && IsPositionInTheGridEmpty(i+1,j,k) == kGRID_USED) {
            r->addDoor(kRIGHT);
        } else {
            r->addWindow(kRIGHT);
        }

        if (j != 0 && IsPositionInTheGridEmpty(i,j-1,k) == kGRID_USED) {
            r->addDoor(kFLOOR);
        }

        if (j != grid_y_-1 && IsPositionInTheGridEmpty(i,j+1,k) == kGRID_USED) {
            r->addDoor(kROOF);
        }

        if (k != 0 && IsPositionInTheGridEmpty(i,j,k-1) == kGRID_USED) {
            r->addDoor(kFRONT);
        }

        if (k != grid_z_-1 && IsPositionInTheGridEmpty(i,j,k+1) == kGRID_USED) {
            r->addDoor(kBACK);
        }
        else if (i != grid_x_-1 && IsPositionInTheGridEmpty(i+1,j,k) == kGRID_USED) {
            r->addWindow(kBACK);
        }
    }
}

} // namespace universe

} // namespace nextfloor
