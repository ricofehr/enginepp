/*
* Universe class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/universe.h"

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cilk/cilk.h>

#include "engine/universe/wall.h"
#include "engine/universe/brick.h"
#include "engine/parallell/cl_parallell.h"
#include "engine/parallell/serial_parallell.h"
#include "engine/parallell/cilk_parallell.h"
#include "engine/helpers/proxy_config.h"

namespace engine {
namespace universe {

namespace {
    double kLastTime = 0;
}

/* Constructor */
Universe::Universe()
{
    InitProxyParallell();
    InitRooms();
}

/* Init parallell proxy pointer */
void Universe::InitProxyParallell()
{
    using engine::parallell::EngineParallell;
    /* Get parallell type from config */
    using engine::helpers::ProxyConfig;
    int type_parallell = ProxyConfig::getSetting<int>("parallell");

    switch (type_parallell) {
        case EngineParallell::kPARALLELL_CILK:
            proxy_parallell_ = std::make_unique<engine::parallell::CilkParallell>();
            break;
        case EngineParallell::kPARALLELL_CL:
            proxy_parallell_ = std::make_unique<engine::parallell::CLParallell>();
            break;
        default:
            proxy_parallell_ = std::make_unique<engine::parallell::SerialParallell>();
            break;
    }

    proxy_parallell_->InitCollisionParallell();
}

/* Create Rooms for gl scene */
void Universe::InitRooms()
{
    /* Reset seed */
    srand (time(NULL));

    CreateGLBuffers();
    GenerateRooms();
    ReinitGrid();
    GenerateWalls();
}

/* Create Universe Objects buffers before using them */
void Universe::CreateGLBuffers()
{
    Brick::CreateBuffers();
    Wall::CreateBuffers();
}

/*   Recompute the placement grid for the room
*   TODO: TO Optimize, too many loops
*/
void Universe::ReinitGrid()
{
    std::vector<std::unique_ptr<Room>> ret;

    cilk_for (auto i = 0; i < kGRID_Y; i++) {
        cilk_for (auto j = 0; j < kGRID_X; j++) {
            cilk_for (auto k = 0; k < kGRID_Z; k++) {
                grid_[i][j][k].clear();
            }
        }
    }

    for (auto o = 0; o < rooms_.size();) {
        /* check grid collision */
        engine::geometry::Box border = rooms_[o]->border();
        std::vector<glm::vec3> coords = border.ComputeCoords();
        auto x1 = coords.at(0)[0];
        auto y1 = coords.at(0)[1];
        auto z1 = coords.at(0)[2];
        auto h1 = coords.at(3)[1] - y1;
        auto w1 = coords.at(1)[0] - x1;
        auto d1 = coords.at(4)[2] - z1;

        rooms_[o]->clear_placements();

        auto grid_0 = glm::vec4(0.0f) - glm::vec4(kGRID_X*kGRID_UNIT_X/2, kGRID_Y*kGRID_UNIT_Y/2, kGRID_Z*kGRID_UNIT_Z/2, 0.0f);

        for (auto i = 0; i < kGRID_Y; i++) {
            auto y2 = grid_0[1] + (i+1) * kGRID_UNIT_Y;
            for (auto j = 0; j < kGRID_X; j++) {
                auto x2 = grid_0[0] + j * kGRID_UNIT_X;
                for (auto k = 0; k < kGRID_Z; k++) {
                    auto z2 = grid_0[2] + (k+1) * kGRID_UNIT_Z;
                    auto h2 = -kGRID_UNIT_Y;
                    auto w2 = kGRID_UNIT_X;
                    auto d2 = -kGRID_UNIT_Z;

                    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
                        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                        rooms_[o]->add_placement(i, j, k);
                        grid_[i][j][k].push_back(rooms_[o].get());
                    }
                }
            }
        }

        if (rooms_[o]->get_placements().size() == 0) {
            rooms_.erase(rooms_.begin() + o);
        } else {
            o++;
        }
    }

    for (auto &r : rooms_) {
        auto i = r->get_placements()[0][0];
        auto j = r->get_placements()[0][1];
        auto k = r->get_placements()[0][2];

        if (i != 0 && grid_[i-1][j][k].size() != 0) {
            r->addDoor(Room::kFLOOR);
            grid_[i-1][j][k][0]->addDoor(Room::kROOF);
        }

        if (i != kGRID_Y-1 && grid_[i+1][j][k].size() != 0) {
            r->addDoor(Room::kROOF);
            grid_[i+1][j][k][0]->addDoor(Room::kFLOOR);
        }

        if (j != 0 && grid_[i][j-1][k].size() != 0) {
            r->addDoor(Room::kLEFT);
            grid_[i][j-1][k][0]->addDoor(Room::kRIGHT);
        }

        if (j != kGRID_X-1 && grid_[i][j+1][k].size() != 0) {
            r->addDoor(Room::kRIGHT);
            grid_[i][j+1][k][0]->addDoor(Room::kLEFT);
        }
        else {
            r->addWindow(Room::kRIGHT);
        }


        if (k != 0 && grid_[i][j][k-1].size() != 0) {
            r->addDoor(Room::kFRONT);
            grid_[i][j][k-1][0]->addDoor(Room::kBACK);
        }

        if (k != kGRID_Z-1 && grid_[i][j][k+1].size() != 0) {
            r->addDoor(Room::kBACK);
            grid_[i][j][k+1][0]->addDoor(Room::kFRONT);
        }
        else if (j != kGRID_X-1 && grid_[i][j+1][k].size() != 0) {
            r->addWindow(Room::kBACK);
        }
    }

    using engine::helpers::ProxyConfig;
    if (ProxyConfig::getSetting<int>("debug") > 0)
        DisplayGrid();
}

/* Display Rooms grid for Universe */
void Universe::DisplayGrid()
{
    std::cout << "=== ROOMS GRID ===" << std::endl << std::endl;
    for (auto i = 0; i < kGRID_Y; i++) {
        std::cout << "=== Floor " << i << std::endl;
        for (auto k = 0; k < kGRID_Z; k++) {
            for (auto j = 0; j < kGRID_X; j++) {
                if (grid_[i][j][k].size() >0)
                    std::cout << "  o";
                else
                    std::cout << "  x";
            }

            std::cout << std::endl;
        }
        std::cout << std::endl << std::endl;
    }
}

/* Generates Random rooms into Universe */
void Universe::GenerateRooms()
{
    /* Check objects count into config file */
    using engine::helpers::ProxyConfig;
    auto nbrooms = ProxyConfig::getSetting<int>("rooms_count");

    while (nbrooms--) {
        GenerateRandomRoom();
    }
}

/* Generates a random Room into Universe */
void Universe::GenerateRandomRoom()
{
    /* Entropy value */
    auto r = rand();
    auto s = rand();
    auto t = rand();

    /* Starting location */
    auto grid_0 = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    grid_0 -= glm::vec4(kGRID_X*kGRID_UNIT_X/2, kGRID_Y*kGRID_UNIT_Y/2, kGRID_Z*kGRID_UNIT_Z/2, 0.0f);

    auto i = r;
    auto j = s;
    auto k = t;
    auto cnt = 0;
    /* Generate and place randomly object into rooms grid */
    while (cnt++ < kGRID_Y * kGRID_X * kGRID_Z) {
        auto l = r % kGRID_Y;
        auto m = s % kGRID_X;
        auto n = t % kGRID_Z;

        auto y2 = grid_0[1] + l * kGRID_UNIT_Y + kGRID_UNIT_Y/2;
        auto x2 = grid_0[0] + m * kGRID_UNIT_X + kGRID_UNIT_X/2;
        auto z2 = grid_0[2] + n * kGRID_UNIT_Z + kGRID_UNIT_Z/2;

        /* 2 cases: grid is empty or another room nearest */
        if (rooms_.size() == 0 ||
            (((l != 0 && grid_[l-1][m][n].size() != 0) ||
              (m != 0 && grid_[l][m-1][n].size() != 0) ||
              (n != 0 && grid_[l][m][n-1].size() != 0) ||
              (l != kGRID_Y-1 && grid_[l+1][m][n].size() != 0) ||
              (m != kGRID_X-1 && grid_[l][m+1][n].size() != 0) ||
              (n != kGRID_Z-1 && grid_[l][m][n+1].size() != 0)) &&
             grid_[l][m][n].size() == 0)) {
            Room *obj = GenerateRoom(glm::vec4(x2, y2, z2, 0.0f));
            obj->add_placement(l, m, n);
            grid_[l][m][n].push_back(obj);
            return;
        }

        /* Compute next grid coordinates */
        switch (i % 6) {
        case 0:
            if (++t - k == kGRID_Z) {
                t = k;
                if (++s - j == kGRID_X) {
                    s = j;
                    r++;
                }
            }
            break;

        case 1:
            if (++s - j == kGRID_Z) {
                s = j;
                if (++r - i == kGRID_Y) {
                    r = i;
                    t++;
                }
            }
            break;

        case 2:
            if (++t - k == kGRID_Z) {
                t = k;
                if (++r - i == kGRID_Y) {
                    r = i;
                    s++;
                }
            }
            break;

        case 3:
            if (++s - j == kGRID_X) {
                s = j;
                if (++t - k == kGRID_Z) {
                    t = k;
                    r++;
                }
            }
            break;

        case 4:
            if (++r - i == kGRID_Y) {
                r = i;
                if (++t - k == kGRID_Z) {
                    t = k;
                    s++;
                }
            }
            break;

        case 5:
            if (++r - i == kGRID_Y) {
                r = i;
                if (++s - j == kGRID_X) {
                    s = j;
                    t++;
                }
            }
            break;
        }
    }
}

/* Create Room Object */
Room *Universe::GenerateRoom(glm::vec4 location)
{
    int ind = rooms_.size();

    /* Init Camera for the first room */
    std::unique_ptr<Camera> cam{nullptr};
    if (rooms_.size() == 0) {
        cam = std::make_unique<Camera>(location[0], location[1] + 1.0f, location[2] + 5.0f,
                                       location[0], location[1] + 1.0f, location[2], 0.0f, 1.0f, 0.0f);
        cam_ = cam.get();
    }

    auto room_ptr{std::make_unique<Room>(location, std::move(cam), proxy_parallell_.get())};
    rooms_.push_back(std::move(room_ptr));

    return rooms_[ind].get();
}

/* Generates Walls and Objects  into Room */
void Universe::GenerateWalls()
{
    for (auto &r : rooms_) {
        r->GenerateWalls();
        r->GenerateObjects();
    }
}

/* Compute and draw next hop for the gl scene */
void Universe::NextHop()
{
    using engine::helpers::ProxyConfig;

    /* Detect current room */
    int active_index = -1;
    cilk_for (auto cnt = 0; cnt < rooms_.size(); cnt++) {
        if (rooms_[cnt]->cam() != nullptr)
            active_index = cnt;
    }

    if (active_index == -1)
        return;

    /* Record moving orders for camera */
    rooms_[active_index]->MoveCamera();

    /* Select displayed rooms: all rooms or 2 clipping levels */
    display_rooms_.clear();
    if (ProxyConfig::getSetting<int>("clipping") > 0) {
        auto map_rooms = GetNeighbors(rooms_[active_index].get(), ProxyConfig::getSetting<int>("clipping") % 2);
        map_rooms[rooms_[active_index]->id()] = rooms_[active_index].get();
        for (auto &r : map_rooms)
            display_rooms_.push_back(r.second);
    } else {
        for (auto &r : rooms_)
            display_rooms_.push_back(r.get());
    }

    /* Detect collision in active rooms */
    cilk_for (auto cnt = 0; cnt < display_rooms_.size(); cnt++) {
        display_rooms_[cnt]->DetectCollision(GetOrderNeighbors(display_rooms_[cnt]));
    }

    /* Universe is only ready after 10 hops */
    if (ready()) {
        for (auto &r : display_rooms_) {
            r->Draw(cam_);
        }
    } else {
        ++ready_;
    }

    /* Recompute object grids for each active rooms and transfert boundary objects */
    tbb::mutex hop_mutex;
    std::map<Room*, std::vector<std::unique_ptr<Model3D>>> replace_objs;
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

                    if (o == nullptr)
                        break;
                }

                /* No orphan object */
                if (o != nullptr)
                    objs.first->TransfertObject(std::move(o), true);
            }
        }
    }

    /* GL functions during object generate, then needs serial execution */
    float freq = 0.0f;
    if ((freq = ProxyConfig::getSetting<float>("load_objects_freq")) > 0.0f) {
        double current_time = glfwGetTime();
        for (auto &r : rooms_) {
            if (!r->IsFull() && current_time - kLastTime >= freq)
                r->GenerateRandomObject();
        }

        if (current_time - kLastTime >= freq)
            kLastTime = current_time;
    }
}

/* Return a group of rooms near each other with a deeping level */
std::map<int, Room*> Universe::GetNeighbors(Room *r, int level)
{
    std::map<int, Room*> neighbors;
    Room *tmp;

    auto i = r->get_placements()[0][0];
    auto j = r->get_placements()[0][1];
    auto k = r->get_placements()[0][2];

    if (i != 0 && grid_[i-1][j][k].size() != 0) {
        tmp = grid_[i-1][j][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (i != kGRID_Y-1 && grid_[i+1][j][k].size() != 0) {
        tmp = grid_[i+1][j][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (j != 0 && grid_[i][j-1][k].size() != 0) {
        tmp = grid_[i][j-1][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (j != kGRID_X-1 && grid_[i][j+1][k].size() != 0) {
        tmp = grid_[i][j+1][k][0];
        neighbors[tmp->id()] = tmp;
    }

    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        tmp = grid_[i][j][k-1][0];
        neighbors[tmp->id()] = tmp;
    }

    if (k != kGRID_Z-1 && grid_[i][j][k+1].size() != 0) {
        tmp = grid_[i][j][k+1][0];
        neighbors[tmp->id()] = tmp;
    }

    if (level) {
        std::map<int, Room*> neighbors_recur_merge;
        for (auto &n : neighbors) {
            auto neighbors_recur = GetNeighbors(n.second, level-1);
            neighbors_recur_merge.insert(neighbors_recur.begin(), neighbors_recur.end());
        }
        neighbors.insert(neighbors_recur_merge.begin(), neighbors_recur_merge.end());
    }

    return neighbors;
}

/* Return the 6 rooms (if all exists) near the targetting room (order by common side) */
std::vector<Room*> Universe::GetOrderNeighbors(Room *r)
{
    std::vector<Room*> neighbors(6, nullptr);
    auto i = r->get_placements()[0][0];
    auto j = r->get_placements()[0][1];
    auto k = r->get_placements()[0][2];

    if (i != 0 && grid_[i-1][j][k].size() != 0) {
        neighbors[Room::kFLOOR] = grid_[i-1][j][k][0];
    }

    if (i != kGRID_Y-1 && grid_[i+1][j][k].size() != 0) {
        neighbors[Room::kROOF] = grid_[i+1][j][k][0];
    }

    if (j != 0 && grid_[i][j-1][k].size() != 0) {
        neighbors[Room::kLEFT] = grid_[i][j-1][k][0];
    }

    if (j != kGRID_X-1 && grid_[i][j+1][k].size() != 0) {
        neighbors[Room::kRIGHT] = grid_[i][j+1][k][0];
    }

    if (k != 0 && grid_[i][j][k-1].size() != 0) {
        neighbors[Room::kFRONT] = grid_[i][j][k-1][0];
    }

    if (k != kGRID_Z-1 && grid_[i][j][k+1].size() != 0) {
        neighbors[Room::kBACK] = grid_[i][j][k+1][0];
    }

    return neighbors;
}

}//namespace universe
}//namespace engine
