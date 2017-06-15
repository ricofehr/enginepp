/**
 *  Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "engine/universe/universe.h"

#include <GLFW/glfw3.h>
#include <map>
#include <iostream>
#include <cilk/cilk.h>

#include "engine/graphics/border.h"
#include "engine/universe/room.h"
#include "engine/universe/random_universe_factory.h"
#include "engine/core/config_engine.h"

namespace engine {

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

    using engine::graphics::Border;
    border_ = std::make_unique<Border>(glm::vec3(grid_x_*grid_unit_x_/2,
                                                 grid_y_*grid_unit_y_/2,
                                                 grid_z_*grid_unit_z_/2),
                                       glm::vec4(0,0,0,0));
}

void Universe::InitDoorsForRooms() noexcept
{
    for (auto &o : objects_) {
        /* Objects contained in Universe are Rooms, with Doors and Windows */
        auto r = dynamic_cast<Room*>(o.get());

        auto i = r->placements()[0][0];
        auto j = r->placements()[0][1];
        auto k = r->placements()[0][2];

        if (i != 0 && grid_[i-1][j][k].size() != 0) {
            r->addDoor(kLEFT);
        }

        if (i != grid_x_-1 && grid_[i+1][j][k].size() != 0) {
            r->addDoor(kRIGHT);
        } else {
            r->addWindow(kRIGHT);
        }

        if (j != 0 && grid_[i][j-1][k].size() != 0) {
            r->addDoor(kFLOOR);
        }

        if (j != grid_y_-1 && grid_[i][j+1][k].size() != 0) {
            r->addDoor(kROOF);
        }

        if (k != 0 && grid_[i][j][k-1].size() != 0) {
            r->addDoor(kFRONT);
        }

        if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
            r->addDoor(kBACK);
        }
        else if (i != grid_x_-1 && grid_[i+1][j][k].size() != 0) {
            r->addWindow(kBACK);
        }
    }
}

void Universe::Draw() noexcept
{
    using engine::core::ConfigEngine;
    auto clipping = ConfigEngine::getSetting<int>("clipping");

    /* Detect current room */
    int active_index = -1;
    cilk_for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (objects_[cnt]->get_camera() != nullptr) {
            active_index = cnt;
        }
    }

    /* if no active Room, return */
    if (active_index == -1) {
        return;
    }

    /* Record moving orders for camera */
    objects_[active_index]->RecordHID();

    /* Select displayed rooms: all rooms or 2 clipping levels */
    if (clipping > 0) {
        display_rooms_.clear();
        display_rooms_ = objects_[active_index]->FindClippingNeighbors(clipping);
        display_rooms_.push_back(objects_[active_index].get());

        /* Add deeping neighbors if clipping */
    } else if(display_rooms_.size() == 0) {
        for (auto &o : objects_) {
            display_rooms_.push_back(o.get());
        }
    }

    /* Detect collision in active rooms */
    cilk_for (auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
        display_rooms_[cnt]->DetectCollision();
    }

    /* Universe is only ready after 10 hops */
    if (ready()) {
        /* Compute new coords after current move */
        cilk_for (auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
            display_rooms_[cnt]->Move();
        }

        /* Draw Rooms on Gl Scene */
        for (auto &r : display_rooms_) {
            r->Draw();
        }
    }

    /* GL functions during object generate, then needs serial execution */
    float freq = 0.0f;
    if ((freq = ConfigEngine::getSetting<float>("load_objects_freq")) > 0.0f) {
        RandomUniverseFactory factory;
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

} // namespace universe

} // namespace engine
