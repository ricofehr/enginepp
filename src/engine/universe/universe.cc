/*
 *  Universe class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/universe/universe.h"

#include <GLFW/glfw3.h>
#include <map>
#include <iostream>
#include <cilk/cilk.h>

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
}

std::vector<std::unique_ptr<Model3D>> Universe::ReinitGrid() noexcept
{
    std::vector<std::unique_ptr<Model3D>> ret = Model3D::ReinitGrid();

    for (auto &o : objects_) {
        /* Objects contained in Universe are Rooms, with Doors and Windows */
        auto r = dynamic_cast<Room*>(o.get());

        auto i = r->placements()[0][0];
        auto j = r->placements()[0][1];
        auto k = r->placements()[0][2];

        if (i != 0 && grid_[i-1][j][k].size() != 0) {
            r->addDoor(kFLOOR);
        }

        if (i != grid_y_-1 && grid_[i+1][j][k].size() != 0) {
            r->addDoor(kROOF);
        }

        if (j != 0 && grid_[i][j-1][k].size() != 0) {
            r->addDoor(kLEFT);
        }

        if (j != grid_x_-1 && grid_[i][j+1][k].size() != 0) {
            r->addDoor(kRIGHT);
        }
        else {
            r->addWindow(kRIGHT);
        }

        if (k != 0 && grid_[i][j][k-1].size() != 0) {
            r->addDoor(kFRONT);
        }

        if (k != grid_z_-1 && grid_[i][j][k+1].size() != 0) {
            r->addDoor(kBACK);
        }
        else if (j != grid_x_-1 && grid_[i][j+1][k].size() != 0) {
            r->addWindow(kBACK);
        }
    }

    return ret;
}

void Universe::Draw() noexcept
{
    using engine::core::ConfigEngine;

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
    if (ConfigEngine::getSetting<int>("clipping") > 0) {
        display_rooms_.clear();
        auto map_rooms = GetNeighbors(objects_[active_index].get(),
                                      ConfigEngine::getSetting<int>("clipping") % 2);
        map_rooms[objects_[active_index]->id()] = objects_[active_index].get();
        for (auto &r : map_rooms) {
            display_rooms_.push_back(r.second);
        }
    } else if(display_rooms_.size() == 0) {
        for (auto &o : objects_) {
            display_rooms_.push_back(o.get());
        }
    }

    /* Detect collision in active rooms */
    cilk_for (auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
        display_rooms_[cnt]->DetectCollision(GetOrderNeighbors(display_rooms_[cnt]));
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

    /* Recompute object grids for each active rooms and transfert boundary objects */
    tbb::mutex hop_mutex;
    std::map<Model3D*, std::vector<std::unique_ptr<Model3D>>> replace_objs;
    cilk_for (auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
        auto objs = display_rooms_[cnt]->ReinitGrid();
        if (objs.size()) {
            tbb::mutex::scoped_lock lock_map(hop_mutex);
            replace_objs[display_rooms_[cnt]] = std::move(objs);
        }
    }

    /* Transfert boundary objects onto nearest room */
    for (auto &objs : replace_objs) {
        if (objs.second.size()) {
            for (auto &o : objs.second) {
                for (auto &r : GetNeighbors(objs.first, 0)) {
                    o = r.second->TransfertObject(std::move(o), false);
                    if (o == nullptr) {
                        break;
                    }
                }

                /* No orphan object */
                if (o != nullptr) {
                    objs.first->TransfertObject(std::move(o), true);
                }
            }
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
