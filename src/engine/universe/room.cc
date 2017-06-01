/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

#include <iostream>

#include "engine/core/config_engine.h"

namespace engine {
namespace universe {

/* Constructors */
Room::Room()
     :Room(glm::vec4(1.0f)) {}

Room::Room(glm::vec4 location)
     :Room(location, std::vector<bool>(6, false), std::vector<bool>(6, false)){}

Room::Room(glm::vec4 location, std::vector<bool> is_doors, std::vector<bool> is_windows) {
    type_ = kMODEL3D_ROOM;
    doors_ = is_doors;
    windows_ = is_windows;

    /* Init Grid Settings */
    grid_x_ = kGRID_X;
    grid_y_ = kGRID_Y;
    grid_z_ = kGRID_Z;
    grid_unit_x_ = kGRID_UNIT_X;
    grid_unit_y_ = kGRID_UNIT_Y;
    grid_unit_z_ = kGRID_UNIT_Z;
    InitGrid();

    using engine::geometry::Box;
    border_ = Box(glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2), location);

    /* Check objects count into config file */
    using engine::core::ConfigEngine;
    nbobjects_ = ConfigEngine::getSetting<int>("objects_count");
}

/* Create 3D walls for Room */
void Room::GenerateWalls() {
    glm::vec3 scale_w = {1.0f, 1.0f, 1.0f};
    glm::vec3 location_0 {0.0f};
    glm::vec4 location_w {0.0f};

    /* First Room 3D Point */
    location_0 = location() - glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2);

    auto it_doors = doors_.begin();
    auto it_windows = windows_.begin();

    /* Top and Roof */
    scale_w = {(grid_x_/8)*grid_unit_x_, grid_unit_y_/4, (grid_z_/8)*grid_unit_z_};
    for (auto i = 0; i < grid_y_+1; i+=(grid_y_)) {
        cilk_for (auto j = 0; j < grid_x_; j+=grid_x_/4) {
            cilk_for (auto k = 0; k < grid_z_; k+=grid_z_/4) {
                /* No brick floor/roof if trapdoor */
                if (!*it_doors || (j != 2*grid_x_/4 && j != 3*grid_x_/4) || (k != 2*grid_x_/4 && k != 3*grid_x_/4)) {
                    tbb::mutex::scoped_lock lock(room_mutex_);
                    location_w = glm::vec4(location_0, 0.0f)
                                 + glm::vec4(scale_w[0], 0.0f, scale_w[2], 0.0f)
                                 + glm::vec4 {j*grid_unit_x_, i*grid_unit_y_, k*grid_unit_z_, 0.0f};
                    if (i == 0) {
                        location_w[1] += grid_unit_y_/4;
                    } else {
                        location_w[1] -= grid_unit_y_/4;
                    }
                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w, (i == 0) ? static_cast<int>(Wall::kTEXTURE_FLOOR): static_cast<int>(Wall::kTEXTURE_TOP))};
                    objects_.push_back(std::move(wall_ptr));
                 }
            }
        }
        ++it_doors;
        ++it_windows;
    }

    scale_w = {grid_unit_x_/4, (grid_y_/6)*grid_unit_y_, (grid_z_/8)*grid_unit_z_};
    for (auto i = 0; i < grid_x_+1; i+=(grid_x_)) {
        cilk_for (auto j = 0; j < grid_y_; j+=grid_y_/3) {
            cilk_for (auto k = 0; k < grid_z_; k+=grid_z_/4) {
                /* No brick wall if Doors or Windows */
                if ((!*it_doors || j > grid_y_/3 || k != 0) &&
                    (!*it_windows || j != grid_y_/3 || (k != 2*grid_z_/4 && k != 3*grid_z_/4))) {
                    tbb::mutex::scoped_lock lock(room_mutex_);
                    location_w = glm::vec4(location_0, 0.0f)
                                 + glm::vec4(0.0f, scale_w[1], scale_w[2], 0.0f)
                                 + glm::vec4 {i*grid_unit_x_, j*grid_unit_y_, k*grid_unit_z_, 0.0f};
                    if (i == 0) {
                        location_w[0] += grid_unit_x_/4;
                    } else {
                        location_w[0] -= grid_unit_x_/4;
                    }
                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w, static_cast<int>(Wall::kTEXTURE_WALL))};
                    objects_.push_back(std::move(wall_ptr));
                }
            }
        }
        ++it_doors;
        ++it_windows;
    }

    scale_w = {(grid_x_/8)*grid_unit_x_, (grid_y_/6)*grid_unit_y_, grid_unit_z_/4};
    for (auto i = 0; i < grid_z_+1; i+=(grid_z_)) {
        cilk_for (auto j = 0; j < grid_y_; j+=grid_y_/3) {
            cilk_for (auto k = 0; k < grid_x_; k+=grid_x_/4) {
                /* No brick wall if Doors or Windows */
                if ((!*it_doors || j > grid_y_/3 || k != 0) &&
                    (!*it_windows || j != grid_y_/3 || (k != 2*grid_x_/4 && k != 3*grid_x_/4))) {
                    tbb::mutex::scoped_lock lock(room_mutex_);
                    location_w = glm::vec4(location_0, 0.0f)
                                 + glm::vec4(scale_w[0], scale_w[1], 0.0f, 0.0f)
                                 + glm::vec4 {k*grid_unit_x_, j*grid_unit_y_, i*grid_unit_z_, 0.0f};
                    if (i == 0) {
                        location_w[2] += grid_unit_z_/4;
                    } else {
                        location_w[2] -= grid_unit_z_/4;
                    }
                    auto wall_ptr{std::make_unique<Wall>(scale_w, location_w, static_cast<int>(Wall::kTEXTURE_WALL))};
                    objects_.push_back(std::move(wall_ptr));
                }
            }
        }
        ++it_doors;
        ++it_windows;
    }
}

/* Generates Random bricks into Room */
void Room::GenerateObjects()
{
    ReinitGrid();

    /* If sequentially object creation, return */
    using engine::core::ConfigEngine;
    if (ConfigEngine::getSetting<float>("load_objects_freq") != 0.0f) {
        return;
    }

    while (nbobjects_) {
        GenerateRandomObject();
    }
}

/* Generates Random bricks into Room */
void Room::GenerateRandomObject()
{
    int index = objects_.size();
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float scale = 1.0f;
    int x0 = 0, z0 = 0;

    /* Entropy value */
    auto r = rand();
    auto s = rand();
    auto t = rand();
    /* For sizes available */
    scale = 1.0f / (float)(index % 3 + 2.0);

    /* 1/7 moves on y axis */
    if (index % 7 == 0) {
        y = (rand() % 20 + 3) * 0.007f;
        x = 0.0f;
        z = 0.0f;
    } else {
        x = (rand() % 20 + 3) * 0.007f;
        x = (r % 2 == 0) ? -x : x;
        z = (rand() % 20 + 3) * 0.007f;
        z = (r % 3 == 0) ? -z : z;
        y = 0.0f;
    }

    /* First 3D point into Room */
    auto grid_0 = location() - glm::vec3(grid_x_*grid_unit_x_/2, grid_y_*grid_unit_y_/2, grid_z_*grid_unit_z_/2);

    auto i = r;
    auto j = s;
    auto k = t;
    auto cnt = 0;
    /* Generate and place randomly object into room grid */
    while (cnt++ < (grid_y_-1) * (grid_x_-1) * (grid_z_-1)) {
        auto l = r % (grid_y_-1);
        auto m = s % (grid_x_-1);
        auto n = t % (grid_z_-1);

        auto y2 = grid_0[1] + l * grid_unit_y_ + grid_unit_y_/2;
        auto x2 = grid_0[0] + m * grid_unit_x_ + grid_unit_x_/2;
        auto z2 = grid_0[2] + n * grid_unit_z_ + grid_unit_z_/2;

        grid_mutex_.lock();
        if (grid_[l][m][n].size() == 0) {
            Model3D *obj = GenerateObject(Model3D::kMODEL3D_BRICK, glm::vec4(x2, y2, z2, 0.0f),
                                          glm::vec4(x, y, z, 0.0f), scale);
            obj->add_placement(l, m, n);

            /* Lock room for ensure only one change at same time on grid_ array */
            grid_[l][m][n].push_back(obj);
            --nbobjects_;
            grid_mutex_.unlock();
            return;
        }
        grid_mutex_.unlock();

        switch (i % 6) {
        case 0:
            if (++t - k == grid_z_) {
                t = k;
                if (++s - j == grid_x_) {
                    s = j;
                    r++;
                }
            }
            break;

        case 1:
            if (++s - j == grid_z_) {
                s = j;
                if (++r - i == grid_y_) {
                    r = i;
                    t++;
                }
            }
            break;

        case 2:
            if (++t - k == grid_z_) {
                t = k;
                if (++r - i == grid_y_) {
                    r = i;
                    s++;
                }
            }
            break;

        case 3:
            if (++s - j == grid_x_) {
                s = j;
                if (++t - k == grid_z_) {
                    t = k;
                    r++;
                }
            }
            break;

        case 4:
            if (++r - i == grid_y_) {
                r = i;
                if (++t - k == grid_z_) {
                    t = k;
                    s++;
                }
            }
            break;

        case 5:
            if (++r - i == grid_y_) {
                r = i;
                if (++s - j == grid_x_) {
                    s = j;
                    t++;
                }
            }
            break;
        }
    }
}

/* Create a moving object into Room */
Model3D *Room::GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale)
{
    int ind = objects_.size();
    std::unique_ptr<Model3D> obj_ptr;

    /* Lock room for ensure only one change at same time on objects_ array */
    tbb::mutex::scoped_lock lock(room_mutex_);

    /* Create Brick object and add to the Current Room */
    if (type_object == Model3D::kMODEL3D_BRICK) {
        obj_ptr = std::make_unique<Brick>(scale, location, move);
    }

    objects_.push_back(std::move(obj_ptr));
    return objects_[ind].get();
}

/* Draw room and all objects inside it */
void Room::Draw(Camera *cam)
{
    /* If current Room, compute new Camera GL Coords */
    if (get_camera() != nullptr) {
        objects_[0]->PrepareDraw(cam);
    }

    /* Compute objects GL coords  */
    cilk_for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (objects_[cnt]->type() != Model3D::kMODEL3D_CAMERA) {
            objects_[cnt]->PrepareDraw(cam);
        }
    }

    /* GL draw in serial (must be not parallell !) */
    for (auto &o: objects_) {
        if (o->type() != Model3D::kMODEL3D_CAMERA) {
            o->Draw();
        }
    }
}

}//namespace universe
}//namespace engine
