/*
 *  RandomUniverseFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 */

#include "engine/universe/random_universe_factory.h"

#include <string>

#include "engine/core/config_engine.h"

namespace engine {

namespace universe {

void RandomUniverseFactory::GenerateBuffers() const
{
    /* Generate Buffers Once */
    Brick::CreateBuffers();
    Wall::CreateBuffers();
}

std::unique_ptr<Universe> RandomUniverseFactory::GenerateUniverse() const
{
    auto uni = std::make_unique<Universe>();

    /* Check objects count into config file */
    using engine::core::ConfigEngine;
    uni->set_missobjects(ConfigEngine::getSetting<int>("rooms_count"));

    std::vector<Room*> rooms;
    while (!uni->IsFull()) {
        rooms.push_back(GenerateRoom(uni.get()));
    }

    /* Init Doors for Rooms */
    uni->InitDoorsForRooms();

    /* Display Universe Grid if Standard Debug mode */
    if (ConfigEngine::getSetting<int>("debug") > ConfigEngine::kDEBUG_TEST) {
        uni->DisplayGrid();
    }

    for (auto &room : rooms) {
        /* Generate Walls of Room */
        GenerateWalls(room);

        /* If sequentially object creation, return */
        using engine::core::ConfigEngine;
        if (ConfigEngine::getSetting<float>("load_objects_freq") == 0.0f) {
            while (!room->IsFull()) {
                GenerateBrick(room);
            }
        }

        /* Display Room Grid only if Full Debug mode */
        if (ConfigEngine::getSetting<int>("debug") >= ConfigEngine::kDEBUG_ALL) {
            room->DisplayGrid();
        }
    }

    /* Compute Recursively Neighbhors for all Objects in Universe */
    uni->ComputeNeighbors();

    return uni;
}

Room* RandomUniverseFactory::GenerateRoom(Universe* uni) const
{
    /* Entropy value */
    auto r = rand();
    auto s = rand();
    auto t = rand();

    /* Init local vars from Room object attributes */
    auto grid_x = uni->gridx();
    auto grid_y = uni->gridy();
    auto grid_z = uni->gridz();
    auto grid_unit_x = uni->grid_unitx();
    auto grid_unit_y = uni->grid_unity();
    auto grid_unit_z = uni->grid_unitz();

    /* Starting location */
    auto grid_0 = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    grid_0 -= glm::vec4(grid_x*grid_unit_x/2, grid_y*grid_unit_y/2, grid_z*grid_unit_z/2, 0.0f);

    auto i = r;
    auto j = s;
    auto k = t;
    auto cnt = 0;

    /* Generate and place randomly object into rooms grid */
    while (cnt++ < grid_x * grid_y * grid_z) {

        auto l = r % grid_x;
        auto m = s % grid_y;
        auto n = t % grid_z;

        auto loc_x = grid_0[0] + l * grid_unit_x + grid_unit_x/2;
        auto loc_y = grid_0[1] + m * grid_unit_y + grid_unit_y/2;
        auto loc_z = grid_0[2] + n * grid_unit_z + grid_unit_z/2;

        /* 2 cases: grid is empty or need another room nearest */
        if (uni->countChilds() == 0 ||
            (uni->IsPositionInTheGridEmpty(l, m, n) == 1 &&
             (!uni->IsPositionInTheGridEmpty(l-1, m, n) ||
              !uni->IsPositionInTheGridEmpty(l, m-1, n) ||
              !uni->IsPositionInTheGridEmpty(l, m, n-1) ||
              !uni->IsPositionInTheGridEmpty(l+1, m, n) ||
              !uni->IsPositionInTheGridEmpty(l, m+1, n) ||
              !uni->IsPositionInTheGridEmpty(l, m, n+1)))) {

            auto location = glm::vec4(loc_x, loc_y, loc_z, 0.0f);
            auto room_ptr{std::make_unique<Room>(location)};

            /* Define moving objects into Room */
            using engine::core::ConfigEngine;
            room_ptr->set_missobjects(ConfigEngine::getSetting<int>("objects_count"));

            /* Init Camera for the first room */
            if (uni->countChilds() == 0) {
                room_ptr->add_child(std::move(GenerateCamera(location)));
            }

            auto room = dynamic_cast<Room*>(uni->add_child(std::move(room_ptr)));

            return room;
        }

        /* Compute next grid coordinates for ensure entropy */
        switch (cnt % 6) {
            case 0:
                if (++t - k == grid_z) {
                    t = k;
                    if (++s - j == grid_y) {
                        s = j;
                        ++r;
                    }
                }
                break;

            case 1:
                if (++s - j == grid_z) {
                    s = j;
                    if (++r - i == grid_x) {
                        r = i;
                        ++t;
                    }
                }
                break;

            case 2:
                if (++t - k == grid_z) {
                    t = k;
                    if (++r - i == grid_x) {
                        r = i;
                        ++s;
                    }
                }
                break;

            case 3:
                if (++s - j == grid_y) {
                    s = j;
                    if (++t - k == grid_z) {
                        t = k;
                        ++r;
                    }
                }
                break;

            case 4:
                if (++r - i == grid_x) {
                    r = i;
                    if (++t - k == grid_z) {
                        t = k;
                        ++s;
                    }
                }
                break;
                
            case 5:
                if (++r - i == grid_x) {
                    r = i;
                    if (++s - j == grid_y) {
                        s = j;
                        ++t;
                    }
                }
                break;
        }
    }

    /* Dont find empty square, try again with different entropy */
    return GenerateRoom(uni);
}

void RandomUniverseFactory::GenerateWalls(Room* room) const
{
    glm::vec3 scale_w = {1.0f, 1.0f, 1.0f};
    glm::vec3 location_0 {0.0f};
    glm::vec4 location_w {0.0f};

    /* Init local vars from Room object attributes */
    auto doors = room->doors();
    auto windows = room->windows();
    auto grid_x = room->gridx();
    auto grid_y = room->gridy();
    auto grid_z = room->gridz();
    auto grid_unit_x = room->grid_unitx();
    auto grid_unit_y = room->grid_unity();
    auto grid_unit_z = room->grid_unitz();
    auto side = 0;

    /* First Room 3D Point */
    location_0 = room->location() - glm::vec3(grid_x*grid_unit_x/2, grid_y*grid_unit_y/2, grid_z*grid_unit_z/2);

    /* Top and Roof */
    scale_w = {(grid_x/8)*grid_unit_x, grid_unit_y/4, (grid_z/8)*grid_unit_z};
    for (auto i = 0; i < grid_y+1; i+=(grid_y)) {
        /* Select side */
        if (i == 0) {
            side = Model3D::kFLOOR;
        } else {
            side = Model3D::kROOF;
        }
        cilk_for (auto j = 0; j < grid_x; j+=grid_x/4) {
            cilk_for (auto k = 0; k < grid_z; k+=grid_z/4) {
                /* No brick floor/roof if trapdoor */
                if (!doors[side] ||
                    (j != 2*grid_x/4 && j != 3*grid_x/4) ||
                    (k != 2*grid_x/4 && k != 3*grid_x/4)) {

                    location_w = glm::vec4(location_0, 0.0f)
                    + glm::vec4(scale_w[0], 0.0f, scale_w[2], 0.0f)
                    + glm::vec4 {j*grid_unit_x, i*grid_unit_y, k*grid_unit_z, 0.0f};

                    if (i == 0) {
                        location_w[1] += grid_unit_y/4;
                    } else {
                        location_w[1] -= grid_unit_y/4;
                    }

                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w,
                                                         (i == 0) ? static_cast<int>(Wall::kTEXTURE_FLOOR):
                                                                    static_cast<int>(Wall::kTEXTURE_TOP))};
                    room->add_child(std::move(wall_ptr));
                }
            }
        }
    }

    scale_w = {grid_unit_x/4, (grid_y/6)*grid_unit_y, (grid_z/8)*grid_unit_z};
    for (auto i = 0; i < grid_x+1; i+=(grid_x)) {
        /* Select side */
        if (i == 0) {
            side = Model3D::kLEFT;
        } else {
            side = Model3D::kRIGHT;
        }
        cilk_for (auto j = 0; j < grid_y; j+=grid_y/3) {
            cilk_for (auto k = 0; k < grid_z; k+=grid_z/4) {
                /* No brick wall if Doors or Windows */
                if ((!doors[side] || j > grid_y/3 || k != 0) &&
                    (!windows[side] || (j != grid_y/3) || (k != 2*grid_z/4 && k != 3*grid_z/4))) {

                    location_w = glm::vec4(location_0, 0.0f)
                    + glm::vec4(0.0f, scale_w[1], scale_w[2], 0.0f)
                    + glm::vec4 {i*grid_unit_x, j*grid_unit_y, k*grid_unit_z, 0.0f};

                    if (i == 0) {
                        location_w[0] += grid_unit_x/4;
                    } else {
                        location_w[0] -= grid_unit_x/4;
                    }

                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w, static_cast<int>(Wall::kTEXTURE_WALL))};
                    room->add_child(std::move(wall_ptr));
                }
            }
        }
    }

    scale_w = {(grid_x/8)*grid_unit_x, (grid_y/6)*grid_unit_y, grid_unit_z/4};
    for (auto i = 0; i < grid_z+1; i+=(grid_z)) {
        /* Select side */
        if (i == 0) {
            side = Model3D::kFRONT;
        } else {
            side = Model3D::kBACK;
        }
        cilk_for (auto j = 0; j < grid_y; j+=grid_y/3) {
            cilk_for (auto k = 0; k < grid_x; k+=grid_x/4) {
                /* No brick wall if Doors or Windows */
                if ((!doors[side] || j > grid_y/3 || k != 0) &&
                    (!windows[side] || j != grid_y/3 || (k != 2*grid_x/4 && k != 3*grid_x/4))) {

                    location_w = glm::vec4(location_0, 0.0f)
                    + glm::vec4(scale_w[0], scale_w[1], 0.0f, 0.0f)
                    + glm::vec4 {k*grid_unit_x, j*grid_unit_y, i*grid_unit_z, 0.0f};

                    if (i == 0) {
                        location_w[2] += grid_unit_z/4;
                    } else {
                        location_w[2] -= grid_unit_z/4;
                    }

                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w, static_cast<int>(Wall::kTEXTURE_WALL))};
                    room->add_child(std::move(wall_ptr));
                }
            }
        }
    }
}

void RandomUniverseFactory::GenerateBrick(Room* room) const
{
    float move_x = 0.0f, move_y = 0.0f, move_z = 0.0f;
    float scale = 1.0f;

    /* Init local vars from Room object attributes */
    auto index = room->countChilds();
    auto grid_x = room->gridx();
    auto grid_y = room->gridy();
    auto grid_z = room->gridz();
    auto grid_unit_x = room->grid_unitx();
    auto grid_unit_y = room->grid_unity();
    auto grid_unit_z = room->grid_unitz();

    /* Entropy value */
    auto r = rand();
    auto s = rand();
    auto t = rand();
    /* For sizes available */
    scale = 1.0f / (float)(index % 3 + 2.0);

    /* 1/7 moves on y axis */
    if (index % 7 == 0) {
        move_y = (rand() % 20 + 3) * 0.007f;
        move_x = 0.0f;
        move_z = 0.0f;
    } else {
        move_x = (rand() % 20 + 3) * 0.007f;
        move_x = (r % 2 == 0) ? -move_x : move_x;
        move_z = (rand() % 20 + 3) * 0.007f;
        move_z = (r % 3 == 0) ? -move_z : move_z;
        move_y = 0.0f;
    }

    /* First 3D point into Room */
    auto grid_0 = room->location() - glm::vec3(grid_x*grid_unit_x/2, grid_y*grid_unit_y/2, grid_z*grid_unit_z/2);

    auto i = r;
    auto j = s;
    auto k = t;
    auto cnt = 0;
    /* Generate and place randomly object into room grid 
       1 grid_unit is reserved every 6 sides (2 in each axe) for Walls */
    while (cnt++ < (grid_x-2) * (grid_y-2) * (grid_z-2)) {
        /* Placements Coords */
        auto l = 1 + r % (grid_x-2);
        auto m = 1 + s % (grid_y-2);
        auto n = 1 + t % (grid_z-2);

        /* Location Coords, center (grid_unit/2) of the grid square */
        auto loc_x = grid_0[0] + l * grid_unit_x + grid_unit_x/2;
        auto loc_y = grid_0[1] + m * grid_unit_y + grid_unit_y/2;
        auto loc_z = grid_0[2] + n * grid_unit_z + grid_unit_z/2;

        if (room->IsPositionInTheGridEmpty(l,m,n) == 1) {
            auto obj = std::make_unique<Brick>(scale,
                                               glm::vec4(loc_x, loc_y, loc_z, 0.0f),
                                               glm::vec4(move_x, move_y, move_z, 0.0f));

            /* Lock room for ensure only one change at same time on grid_ array */
            room->add_child(std::move(obj));

            return;
        }

        /* Ensure entropy for random placement */
        switch (cnt % 6) {
            case 0:
                if (++t - k == (grid_z-2)) {
                    t = k;
                    if (++s - j == (grid_y-2)) {
                        s = j;
                        ++r;
                    }
                }
                break;

            case 1:
                if (++s - j == (grid_y-2)) {
                    s = j;
                    if (++r - i == (grid_x-2)) {
                        r = i;
                        ++t;
                    }
                }
                break;

            case 2:
                if (++t - k == (grid_z-2)) {
                    t = k;
                    if (++r - i == (grid_x-2)) {
                        r = i;
                        ++s;
                    }
                }
                break;

            case 3:
                if (++s - j == (grid_y-2)) {
                    s = j;
                    if (++t - k == (grid_z-2)) {
                        t = k;
                        ++r;
                    }
                }
                break;

            case 4:
                if (++r - i == (grid_x-2)) {
                    r = i;
                    if (++t - k == (grid_z-2)) {
                        t = k;
                        ++s;
                    }
                }
                break;
                
            case 5:
                if (++r - i == (grid_x-2)) {
                    r = i;
                    if (++s - j == (grid_y-2)) {
                        s = j;
                        ++t;
                    }
                }
                break;
        }
    }

    /* Dont find empty square, try again with different entropy */
    return GenerateBrick(room);
}

std::unique_ptr<Camera> RandomUniverseFactory::GenerateCamera(glm::vec3 location) const
{
    return std::make_unique<Camera>(location[0],
                                    location[1] + 1.0f,
                                    location[2] + 5.0f,
                                    location[0],
                                    location[1] + 1.0f,
                                    location[2],
                                    0.0f, 1.0f, 0.0f);
}

} // universe

} // engine
