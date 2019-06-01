/**
 *  @file demo_universe_factory.cc
 *  @brief DemoUniverseFactory class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/universe/factory/demo_universe_factory.h"

#include <string>
#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace universe {

namespace factory {

std::unique_ptr<nextfloor::universe::Universe> DemoUniverseFactory::GenerateUniverse() const
{
    using nextfloor::core::CommonServices;

    auto uni = std::make_unique<nextfloor::universe::Universe>();
    uni->set_missobjects(4);

    std::vector<Room*> rooms;
    while (!uni->IsFull()) {
        rooms.push_back(GenerateRoom(uni.get()));
    }

    uni->InitDoorsForRooms();
    uni->DisplayGrid();

    for (auto &room : rooms) {
        GenerateWalls(room);
        while (!room->IsFull()) {
            GenerateBrick(room);
        }
        room->DisplayGrid();
    }

    return uni;
}

nextfloor::universe::Room* DemoUniverseFactory::GenerateRoom(nextfloor::universe::Universe* universe) const
{
    auto grid_0 = universe->ComputeFirstPointInGrid();

    assert(universe->countChilds() < 4);

    /* Romm placement into square grid */
    float l = int(universe->countChilds() >> 1) + 0.5f;
    float n = universe->countChilds() % 2 + 0.5f;
    auto loc_x = grid_0[0] + l * universe->grid_unitx();
    auto loc_z = grid_0[2] + n * universe->grid_unitz();
    auto location = glm::vec4(loc_x, grid_0[1], loc_z, 0.0f);

    auto room_ptr{std::make_unique<Room>(location)};
    /* Define 6 moving objects into Room */
    room_ptr->set_missobjects(6);

    /* Init Camera for the first room */
    if (universe->countChilds() == 0) {
        room_ptr->add_child(std::move(GenerateCamera(location)));
    }

    auto room = dynamic_cast<Room*>(universe->add_child(std::move(room_ptr)));
    return room;
}

void DemoUniverseFactory::GenerateWalls(nextfloor::universe::Room* room) const
{
    GenerateWallYFixed(room, Model3D::kFLOOR);
    GenerateWallYFixed(room, Model3D::kROOF);
    GenerateWallXFixed(room, Model3D::kLEFT);
    GenerateWallXFixed(room, Model3D::kRIGHT);
    GenerateWallZFixed(room, Model3D::kFRONT);
    GenerateWallZFixed(room, Model3D::kBACK);
}

/* TODO: refacto, because too much things,less of clarity, and redundancy with the 2 other ones */
void DemoUniverseFactory::GenerateWallYFixed(nextfloor::universe::Room* room, int side) const
{
    using nextfloor::universe::stationary::Wall;

    glm::vec3 scale_w = {1.0f, 1.0f, 1.0f};
    glm::vec3 location_0 {0.0f};

    /* Init local vars from Room object attributes */
    auto doors = room->doors();
    auto windows = room->windows();
    auto grid_x = room->gridx();
    auto grid_y = room->gridy();
    auto grid_z = room->gridz();
    auto grid_unit_x = room->grid_unitx();
    auto grid_unit_y = room->grid_unity();
    auto grid_unit_z = room->grid_unitz();

    auto i = 0;
    if (side == Model3D::kROOF) {
        i = grid_y;
    }
    location_0 = room->ComputeFirstPointInGrid();
    scale_w = {(grid_x/8)*grid_unit_x, grid_unit_y/4, (grid_z/8)*grid_unit_z};

    tbb::parallel_for (0, grid_x, grid_x/4, [&](int j) {
        tbb::parallel_for (0, grid_z, grid_z/4, [&](int k) {
            /* No brick floor/roof if trapdoor */
            if (!doors[side] ||
                (j != 2*grid_x/4 && j != 3*grid_x/4) ||
                (k != 2*grid_z/4 && k != 3*grid_z/4)) {

                auto location_w = glm::vec4(location_0, 0.0f)
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
        });
    });
}

/* TODO: refacto, because too much things,less of clarity, and redundancy with the 2 other ones */
void DemoUniverseFactory::GenerateWallXFixed(nextfloor::universe::Room* room, int side) const
{
    using nextfloor::universe::stationary::Wall;

    glm::vec3 scale_w = {1.0f, 1.0f, 1.0f};
    glm::vec3 location_0 {0.0f};

    /* Init local vars from Room object attributes */
    auto doors = room->doors();
    auto windows = room->windows();
    auto grid_x = room->gridx();
    auto grid_y = room->gridy();
    auto grid_z = room->gridz();
    auto grid_unit_x = room->grid_unitx();
    auto grid_unit_y = room->grid_unity();
    auto grid_unit_z = room->grid_unitz();

    auto i = 0;
    if (side == Model3D::kRIGHT) {
        i = grid_x;
    }
    location_0 = room->ComputeFirstPointInGrid();
    scale_w = {grid_unit_x/4, (grid_y/6)*grid_unit_y, (grid_z/8)*grid_unit_z};

    tbb::parallel_for (0, grid_y, grid_y/3, [&](int j) {
        tbb::parallel_for (0, grid_z, grid_z/4, [&](int k) {
            /* No brick wall if Doors or Windows */
            if ((!doors[side] || j > grid_y/3 || k != 0) &&
                (!windows[side] || (j != grid_y/3) || (k != 2*grid_z/4 && k != 3*grid_z/4))) {

                auto location_w = glm::vec4(location_0, 0.0f)
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
        });
    });
}

/* TODO: refacto, because too much things,less of clarity, and redundancy with the 2 other ones */
void DemoUniverseFactory::GenerateWallZFixed(nextfloor::universe::Room* room, int side) const
{
    using nextfloor::universe::stationary::Wall;

    glm::vec3 scale_w = {1.0f, 1.0f, 1.0f};
    glm::vec3 location_0 {0.0f};

    /* Init local vars from Room object attributes */
    auto doors = room->doors();
    auto windows = room->windows();
    auto grid_x = room->gridx();
    auto grid_y = room->gridy();
    auto grid_z = room->gridz();
    auto grid_unit_x = room->grid_unitx();
    auto grid_unit_y = room->grid_unity();
    auto grid_unit_z = room->grid_unitz();

    auto i = 0;
    if (side == Model3D::kBACK) {
        i = grid_z;
    }
    location_0 = room->ComputeFirstPointInGrid();
    scale_w = {(grid_x/8)*grid_unit_x, (grid_y/6)*grid_unit_y, grid_unit_z/4};

    tbb::parallel_for (0, grid_y, grid_y/3, [&](int j) {
        tbb::parallel_for (0, grid_x, grid_x/4, [&](int k) {
            /* No brick wall if Doors or Windows */
            if ((!doors[side] || j > grid_y/3 || k != 0) &&
                (!windows[side] || j != grid_y/3 || (k != 2*grid_x/4 && k != 3*grid_x/4))) {

                auto location_w = glm::vec4(location_0, 0.0f)
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
        });
    });
}

/* TODO: refacto, because too much things,less of clarity, and redundancy */
void DemoUniverseFactory::GenerateBrick(nextfloor::universe::Room* room) const
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
    using nextfloor::core::CommonServices;
    auto r = CommonServices::getRandomGenerator()->GenerateNumber();
    auto s = CommonServices::getRandomGenerator()->GenerateNumber();
    auto t = CommonServices::getRandomGenerator()->GenerateNumber();
    /* For sizes available */
    scale = 1.0f / (float)(index % 4 + 2.0);

    /* 1/7 moves on y axis */
    if (index % 7 == 0) {
        move_y = (CommonServices::getRandomGenerator()->GenerateNumber() % 20 + 3) * 0.0045f;
        move_x = 0.0f;
        move_z = 0.0f;
    } else {
        move_x = (CommonServices::getRandomGenerator()->GenerateNumber() % 20 + 3) * 0.0045f;
        move_x = (r % 2 == 0) ? -move_x : move_x;
        move_z = (CommonServices::getRandomGenerator()->GenerateNumber() % 20 + 3) * 0.0045f;
        move_z = (r % 3 == 0) ? -move_z : move_z;
        move_y = 0.0f;
    }

    /* First 3D point into Room */
    auto grid_0 = room->ComputeFirstPointInGrid();

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
            using nextfloor::universe::dynamic::Brick;
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

std::unique_ptr<dynamic::Camera> DemoUniverseFactory::GenerateCamera(glm::vec4 location) const
{
    using nextfloor::universe::dynamic::Camera;

    return std::make_unique<Camera>(location[0],
                                    location[1] + 1.0f,
                                    location[2] + 5.0f,
                                    location[0],
                                    location[1] + 1.0f,
                                    location[2],
                                    0.0f, 1.0f, 0.0f);
}

void DemoUniverseFactory::GenerateBuffers() const
{
    /* Generate Buffers Once */
    dynamic::Brick::CreateBuffers();
    stationary::Wall::CreateBuffers();
}

} // namespace factory

} // universe

} // engine
