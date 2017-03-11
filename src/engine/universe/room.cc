/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

#include <unistd.h>

#include <iostream>
#include <cilk/cilk.h>


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

    GenerateObjects();
}

/* Generates Random bricks into Room */
void Room::GenerateObjects() {
    int r;
    float x, y, z;
    float scale;
    int x0, z0;
    tbb::mutex genobjects_mutex;

    srand(time(NULL));
    /* Parallell objects generation with cilkplus */
    cilk_for (auto i = 0; i < nbobjects_; i++) {
        /* Entropy value */
        r = rand();
        /* For sizes available for brick */
        scale = 1.0f / (float)(i % 4 + 1.0);

        /* 1/10 of bricks are static */
        if (i % 10 == 0) {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }
        /* One brick every 8 moves on y axis */
        if (i % 7 == 0) {
            y = rand() % 30 * 0.01f + 0.3f;
            x = 0.0f;
            z = 0.0f;
        } else {
            x = rand() % 70 * 0.01f + 0.04;
            x = (r % 2 == 0) ? -x : x;
            z = rand() % 70 * 0.01f + 0.04f;
            z = (r % 3 == 0) ? -z : z;
            y = 0.0f;
        }

        /* Initial x and z coordinates */
        x0 = rand() % 10;
        x0 = (x0 % 2 == 0) ? -x0 : x0;
        z0 = rand() % 15;
        z0 = (z0 % 2 == 0) ? -z0 : z0;

        /* Create Brick object and add to the Current Room */
        auto brick_ptr{std::make_unique<Brick>(scale,
                                       location_ + glm::vec4(x0, -4.8f + (r%8), z0, 0.0f),
                                       glm::vec4(x * 0.3f, y * 0.3f, z * 0.3f, 0.0f))};

        /* Ensure that only one push_back at same time */
        {
            tbb::mutex::scoped_lock lock(genobjects_mutex);
            objects_.push_back(std::move(brick_ptr));
        }
    }

}

/* Draw room and all objects inside it */
void Room::Draw() {
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
    /* Record moving orders for camera */
    cam_->Move();
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
            recompute = object->DetectCollision(room_objects[i], collision_mutex_, proxy_parallell_);
            for (auto &r : recompute) {
                cilk_spawn PivotCollision(r);
            }
        }
    }

    cilk_sync;
}

}//namespace universe
}//namespace engine
