/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

#include <unistd.h>

#include <iostream>
#include <unistd.h>
#include <cilk/cilk.h>

#include "engine/helpers/proxy_config.h"

namespace engine {
namespace universe {

/* Constructors */
Room::Room()
     :Room(glm::vec4(1.0f)) {}

Room::Room(glm::vec4 location)
     :Room(location, {false, false, false, false}, nullptr, nullptr){}

Room::Room(glm::vec4 location, std::vector<bool> is_doors, Camera *cam,
           engine::parallell::EngineParallell *proxy_parallell) {
    location_ = location;
    cam_ = cam;
    proxy_parallell_ = proxy_parallell;
    /* 4 walls, floor and roof */
    for (auto face = 0; face < 6; face++) {
        auto wall_ptr{std::make_unique<Wall>(face, 15.0f, location_)};
        walls_.push_back(std::move(wall_ptr));
    }

    glm::vec4 location_center_face[4] = {
        {0.0f, 0.0f, 14.95f, 0.0f},
        {14.95f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, -14.95f, 0.0f},
        {-14.95f, 0.0f, 0.0f, 0.0f}
    };

    /* On each room, 2 doors and 2 windows (windows are on face with no door) */
    for (auto face = 0; face < 4; face++) {
        if (is_doors[face]) {
            auto door_ptr{std::make_unique<Door>(face, 15.0f, location_ + location_center_face[face])};
            doors_.push_back(std::move(door_ptr));
        } else {
            auto window_ptr{std::make_unique<WindowModel>(face, 15.0f, location_ + location_center_face[face])};
            windows_.push_back(std::move(window_ptr));
        }
    }

    /* Check objects count into config file */
    using engine::helpers::ProxyConfig;
    nbobjects_ = ProxyConfig::getSetting<int>("objects_count");

    ReinitGrid();
    if (!ProxyConfig::getSetting<bool>("load_objects_seq"))
        GenerateObjects();
}

/*
*   Recompute the placement grid for the room
*   TODO: TO Optimize, too many loops
*/
void Room::ReinitGrid()
{
    cilk_for (auto i = 0; i < 5; i++) {
        cilk_for (auto j = 0; j < 15; j++) {
            cilk_for (auto k = 0; k < 15; k++) {
                grid_[i][j][k] = false;
            }
        }
    }

    /* Prepare an unique objects collection for grid positions */
    auto cnt = 0;
    std::vector<Model3D*> room_objects(1 + objects_.size());
    room_objects[cnt++] = cam_;
    for (auto &obj : objects_)
        room_objects[cnt++] = obj.get();

    cilk_for (auto o = 0; o < room_objects.size(); o++) {
        /* check grid collision */
        engine::geometry::Box border = room_objects[o]->border();
        std::vector<glm::vec3> coords = border.ComputeCoords();
        auto x1 = coords.at(0)[0];
        auto y1 = coords.at(0)[1];
        auto z1 = coords.at(0)[2];
        auto h1 = coords.at(3)[1] - y1;
        auto w1 = coords.at(1)[0] - x1;
        auto d1 = coords.at(4)[2] - z1;

        cilk_for (auto i = 0; i < 5; i++) {
            auto y2 = (-5.0f + i*2) + location_[1] + 2.0f;
            cilk_for (auto j = 0; j < 15; j++) {
                auto x2 = (-15.0f + j*2) + location_[0];
                cilk_for (auto k = 0; k < 15; k++) {
                    auto z2 = (-15.0f + k*2) + location_[2] + 2.0f;
                    auto h2 = -2.0f;
                    auto w2 = 2.0f;
                    auto d2 = -2.0f;

                    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
                        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                        tbb::mutex::scoped_lock lock(room_mutex_);
                        grid_[i][j][k] = true;
                    }
                }
            }
        }
    }
}

/* Generates Random bricks into Room */
void Room::GenerateObjects()
{
    /* Reset seed */
    srand (time(NULL));

    /* Parallell objects generation with cilkplus */
    for (auto i = 0; i < nbobjects_; i++) {
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
    scale = 1.0f / (float)(index % 4 + 1.0);

    /* 1/10 are static */
    if (index % 10 == 0) {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    }
    /* 1/7 moves on y axis */
    else if (index % 7 == 0) {
        y = (rand() % 20 + 3) * 0.01f;
        x = 0.0f;
        z = 0.0f;
    } else {
        x = (rand() % 20 + 3) * 0.01f;
        x = (r % 2 == 0) ? -x : x;
        z = (rand() % 20 + 3) * 0.01f;
        z = (r % 3 == 0) ? -z : z;
        y = 0.0f;
    }

    /* Initial x and z coordinates */
    x0 = rand() % 10;
    x0 = (x0 % 2 == 0) ? -x0 : x0;
    z0 = rand() % 15;
    z0 = (z0 % 2 == 0) ? -z0 : z0;

    /* Generate and place randomly object into room grid */
    for (auto i = 0; i < 5; i++) {
        auto l = r % 5;
        auto y2 = (-5.0f + l*2) + location_[1] + 1.0f;
        for (auto j = 0; j < 15; j++) {
           auto m = s % 15;
           auto x2 = (-15.0f + m*2) + location_[0] + 1.0f;
           for (auto k = 0; k < 15; k++) {
               auto n = t % 15;
               auto z2 = (-15.0f + n*2) + location_[2] + 1.0f;
               if (!grid_[l][m][n]) {
                   grid_[l][m][n] = true;
                   GenerateObject(Model3D::kMODEL3D_BRICK, glm::vec4(x2, y2, z2, 0.0f),
                                  glm::vec4(x, y, z, 0.0f), scale);
                   return;
                }
            t++;
            }
        s++;
        }
    r++;
    }
}

void Room::GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale)
{
    std::unique_ptr<Model3D> obj_ptr;

    /* Create Brick object and add to the Current Room */
    if (type_object == Model3D::kMODEL3D_BRICK)
        obj_ptr = std::make_unique<Brick>(scale, location_ + location, move);

    objects_.push_back(std::move(obj_ptr));
}

/* Draw room and all objects inside it */
void Room::Draw()
{
    /* Prepare an unique objects collection for collision detection */
    auto cnt = 0;
    std::vector<Model3D*> room_objects(walls_.size() + windows_.size() + doors_.size() + objects_.size());
    for (auto &o : walls_)
        room_objects[cnt++] = o.get();
    for (auto &o : windows_)
        room_objects[cnt++] = o.get();
    for (auto &o : doors_)
        room_objects[cnt++] = o.get();
    for (auto &o : objects_)
        room_objects[cnt++] = o.get();

    /* Parallell compute new coords and init new move vector */
    cilk_for (auto cnt = 0; cnt < room_objects.size(); cnt++) {
        room_objects[cnt]->PrepareDraw(cam_);
    }

    /* GL draw in serial (not threaded !) */
    for (auto &o: room_objects)
        o->Draw();
}

/* Detect collisions inside current room */
void Room::DetectCollision()
{
    /* First check camera collision */
    if (cam_->IsMoved()) {
        PivotCollision(cam_);
    }

    /* For all others moving objects
       Parallell loop with cilkplus */
    cilk_for (auto i = 0; i < objects_.size(); i++) {
        if (objects_[i]->IsMoved())
            PivotCollision(objects_[i].get());
    }
}

/* Detect all collision for one Object */
void Room::PivotCollision(Model3D *object)
{
    /* If collisionsi, previously computed, are cancelled for an object,
       we need to detect again for this one */
    std::vector<Model3D*> recompute;

    /* Prepare an unique objects collection for collision detection */
    auto cnt = 0;
    std::vector<Model3D*> room_objects(1 + walls_.size() + windows_.size() + doors_.size() + objects_.size());
    room_objects[cnt++] = cam_;
    for (auto &o : walls_)
        room_objects[cnt++] = o.get();
    for (auto &o : windows_)
        room_objects[cnt++] = o.get();
    for (auto &o : doors_)
        room_objects[cnt++] = o.get();
    for (auto &o : objects_)
        room_objects[cnt++] = o.get();

    /* Parallell collision loop for objects with cilkplus */
    cilk_for (auto i = 0; i < room_objects.size(); i++) {
        if (*object != *room_objects[i]) {
            recompute = object->DetectCollision(room_objects[i], room_mutex_, proxy_parallell_);
            for (auto &r : recompute) {
                cilk_spawn PivotCollision(r);
            }
        }
    }

    cilk_sync;
}

}//namespace universe
}//namespace engine
