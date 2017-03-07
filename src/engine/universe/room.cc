/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

#include <unistd.h>

#include <iostream>

#include "engine/universe/camera.h"

namespace engine {

namespace helpers {
namespace proxygl {
    extern engine::universe::Camera* kCam;
}//namespace proxygl
}//namespace helpers

namespace universe {

/* Constructors */
Room::Room()
     :Room(glm::vec4(1.0f)) {}

Room::Room(glm::vec4 location)
     :Room(location, {false, false, false, false}, nullptr){}

Room::Room(glm::vec4 location, std::vector<bool> is_doors, engine::parallell::EngineParallell *proxy_parallell) {
    location_ = location;
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

    srand(time(NULL));
    for (auto i = 0; i < 45; i++) {
        /* Entropy value */
        r = rand();
        /* For sizes available for brick */
        scale = 1.0f / (float)(i % 4 + 1.0);

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
        objects_.push_back(std::move(brick_ptr));
    }

}

/* Draw room and all objects inside it */
void Room::Draw() {
    /* Show all faces of walls, windows and doors */
    glDisable(GL_CULL_FACE);

    for (auto &w : walls_) {
        w->Draw();
    }

    for (auto &d : doors_) {
        d->Draw();
    }

    for (auto &w : windows_) {
        w->Draw();
    }

    /* Optimize display for moving objects */
    glEnable(GL_CULL_FACE);

    for (auto &o : objects_) {
        o->Draw();
    }
}

/* Detect collisions inside current room */
void Room::DetectCollision()
{
    using engine::helpers::proxygl::kCam;

    /* Record moving orders for camera */
    kCam->Move();
    /* First check camera collision */
    if (kCam->IsMoved()) {
        PivotCollision(kCam);
    }

    /* And for all others moving objects */
    for (auto &o : objects_) {
        PivotCollision(o.get());
    }
}

/* Detect all collision for one Object */
void Room::PivotCollision(Model3D *object)
{
    using engine::helpers::proxygl::kCam;

    /* If collisionsi, previously computed, are cancelled for an object,
       we need to detect again for this one */
    std::vector<Model3D*> recompute;

    if (*object != *kCam) {
        recompute = object->DetectCollision(kCam, proxy_parallell_);
        for (auto &r : recompute) {
            PivotCollision(r);
        }
    }

    for (auto &w : walls_) {
        recompute = object->DetectCollision(w.get(), proxy_parallell_);
        for (auto &r : recompute) {
            PivotCollision(r);
        }
    }

    for (auto &w : windows_) {
        recompute = object->DetectCollision(w.get(), proxy_parallell_);
        for (auto &r : recompute) {
            PivotCollision(r);
        }
    }

    for (auto &d : doors_) {
        recompute = object->DetectCollision(d.get(), proxy_parallell_);
        for (auto &r : recompute) {
            PivotCollision(r);
        }
    }

    for (auto &p : objects_) {
        if (*object == *p)
                continue;

        recompute = object->DetectCollision(p.get(), proxy_parallell_);
        for (auto &r : recompute) {
            PivotCollision(r);
        }
    }
}

}//namespace universe
}//namespace engine
