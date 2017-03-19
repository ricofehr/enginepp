/*
* Room class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/room.h"

#include <unistd.h>

#include <iostream>
#include <map>
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
        auto wall_ptr{std::make_unique<Wall>(face, static_cast<float>(kROOM_SCALE), location_)};
        walls_.push_back(std::move(wall_ptr));
    }

    glm::vec4 location_center_face[4] = {
        {0.0f, 0.0f, (kGRID_Z*kGRID_UNIT/2)-0.05f, 0.0f},
        {(kGRID_X*kGRID_UNIT/2)-0.05f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, -(kGRID_Z*kGRID_UNIT/2)+0.05f, 0.0f},
        {-(kGRID_X*kGRID_UNIT)/2+0.05f, 0.0f, 0.0f, 0.0f}
    };

    /* On each room, 2 doors and 2 windows (windows are on face with no door) */
    for (auto face = 0; face < 4; face++) {
        if (is_doors[face]) {
            auto door_ptr{std::make_unique<Door>(face, static_cast<float>(kROOM_SCALE), location_ + location_center_face[face])};
            doors_.push_back(std::move(door_ptr));
        } else {
            auto window_ptr{std::make_unique<WindowModel>(face, static_cast<float>(kROOM_SCALE), location_ + location_center_face[face])};
            windows_.push_back(std::move(window_ptr));
        }
    }

    /* Check objects count into config file */
    using engine::helpers::ProxyConfig;
    nbobjects_ = ProxyConfig::getSetting<int>("objects_count");

    /* Reset seed */
    srand (time(NULL));

    ReinitGrid();
    if (ProxyConfig::getSetting<float>("load_objects_freq") == 0.0f)
        GenerateObjects();
}

/*
*   Recompute the placement grid for the room
*   TODO: TO Optimize, too many loops
*/
void Room::ReinitGrid()
{
    cilk_for (auto i = 0; i < kGRID_Y; i++) {
        cilk_for (auto j = 0; j < kGRID_X; j++) {
            cilk_for (auto k = 0; k < kGRID_Z; k++) {
                grid_[i][j][k].clear();
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

        room_objects[o]->clear_placements();

        cilk_for (auto i = 0; i < kGRID_Y; i++) {
            auto y2 = (-kGRID_Y + i*2) * kGRID_UNIT/2 + location_[1] + kGRID_UNIT;
            cilk_for (auto j = 0; j < kGRID_X; j++) {
                auto x2 = (-kGRID_X + j*2) * kGRID_UNIT/2 + location_[0];
                cilk_for (auto k = 0; k < kGRID_Z; k++) {
                    auto z2 = (-kGRID_Z + k*2) * kGRID_UNIT/2 + location_[2] + kGRID_UNIT;
                    auto h2 = -kGRID_UNIT;
                    auto w2 = kGRID_UNIT;
                    auto d2 = -kGRID_UNIT;

                    if (x2 < x1 + w1 && x2 + w2 > x1 && y2 + h2 < y1 &&
                        y2 > y1 + h1 && z2 > z1 + d1 && z2 + d2 < z1) {
                        tbb::mutex::scoped_lock lock(room_mutex_);
                        room_objects[o]->add_placement(i, j, k);
                        grid_[i][j][k].push_back(room_objects[o]);
                    }
                }
            }
        }
    }
}

/* Generates Random bricks into Room */
void Room::GenerateObjects()
{
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
    scale = 1.0f / (float)(index % 3 + 2.0);

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

    /* Generate and place randomly object into room grid */
    for (auto i = 0; i < kGRID_Y; i++) {
        auto l = r % kGRID_Y;
        auto y2 = (-kGRID_Y + l*2) * kGRID_UNIT/2 + location_[1] + kGRID_UNIT/2;
        for (auto j = 0; j < kGRID_X; j++) {
           auto m = s % kGRID_X;
           auto x2 = (-kGRID_X + m*2) * kGRID_UNIT/2 + location_[0] + kGRID_UNIT/2;
           for (auto k = 0; k < kGRID_Z; k++) {
               auto n = t % kGRID_Z;
               auto z2 = (-kGRID_Z + n*2) * kGRID_UNIT/2 + location_[2] + kGRID_UNIT/2;
               if (grid_[l][m][n].size() == 0) {
                   Model3D *obj = GenerateObject(Model3D::kMODEL3D_BRICK, glm::vec4(x2, y2, z2, 0.0f),
                                                 glm::vec4(x, y, z, 0.0f), scale);
                   obj->add_placement(l, m, n);
                   grid_[l][m][n].push_back(obj);
                   return;
                }
            t++;
            }
        s++;
        }
    r++;
    }
}

Model3D *Room::GenerateObject(int type_object, glm::vec4 location, glm::vec4 move, float scale)
{
    int ind = objects_.size();
    std::unique_ptr<Model3D> obj_ptr;

    /* Create Brick object and add to the Current Room */
    if (type_object == Model3D::kMODEL3D_BRICK)
        obj_ptr = std::make_unique<Brick>(scale, location_ + location, move);

    objects_.push_back(std::move(obj_ptr));
    return objects_[ind].get();
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
    if (is_active_ &&
        cam_->IsMoved()) {
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
    /* ensure that 2 same objects are not checked at the same time */
    object->lock();

    /* Prepare an unique objects collection for collision detection */
    std::map<int, Model3D*> grid_objects;

    /* Grid coordinates for current object */
    std::vector<std::vector<int>> placements = object->get_placements();

    /* Check all grid placements for current object and select other objects near this one */
    for (auto &p : placements) {
        auto y = p[0];
        auto x = p[1];
        auto z = p[2];
        Model3D *obj = nullptr;

        /* Roof / Top adding */
        if (y == 0 && object->IsMovedY()) {
            obj = walls_[Wall::kWALL_BOTTOM].get();
            grid_objects[obj->id()] = obj;
        }

        if (y == (kGRID_Y-1) && object->IsMovedY()) {
            obj = walls_[Wall::kWALL_TOP].get();
            grid_objects[obj->id()] = obj;
        }

        /* 4 Walls adding */
        if (x == 0 && object->IsMovedX()) {
            obj = walls_[Wall::kWALL_LEFT].get();
            grid_objects[obj->id()] = obj;
            if (*object == *cam_) {
                obj = doors_[Wall::kWALL_LEFT].get();
                grid_objects[obj->id()] = obj;
            }
        }

        if (x == (kGRID_X-1) && object->IsMovedX()) {
            obj = walls_[Wall::kWALL_RIGHT].get();
            grid_objects[obj->id()] = obj;
            if (*object == *cam_) {
                obj = doors_[Wall::kWALL_RIGHT].get();
                grid_objects[obj->id()] = obj;
            }
        }

        if (z == 0 && object->IsMovedZ()) {
            obj = walls_[Wall::kWALL_BACK].get();
            grid_objects[obj->id()] = obj;
            if (*object == *cam_) {
                obj = doors_[Wall::kWALL_BACK].get();
                grid_objects[obj->id()] = obj;
            }
        }

        if (z == (kGRID_Z-1) && object->IsMovedZ()) {
            obj = walls_[Wall::kWALL_FRONT].get();
            grid_objects[obj->id()] = obj;
            if (*object == *cam_) {
                obj = doors_[Wall::kWALL_FRONT].get();
                grid_objects[obj->id()] = obj;
            }
        }

        /* Other room objects adding */
        for (auto i = y-1; i <= y+1; i++) {
            if (i < 0 || i >= kGRID_Y)
                continue;
            for (auto j = x-1; j <= x+1; j++) {
                if (j < 0 || j >= kGRID_X)
                    continue;
                for (auto k = z-1; k <= z+1; k++) {
                    if (k < 0 || k >= kGRID_Z ||
                        grid_[i][j][k].size() == 0)
                        continue;

                    for (auto &obj : grid_[i][j][k]) {
                        if (*obj == *object)
                            continue;
                        grid_objects[obj->id()] = obj;
                    }
                }
            }
        }
    }

    /* Prepare vector for collision compute */
    std::vector<Model3D*> room_objects;
    for (auto & obj_pair : grid_objects)
        room_objects.push_back(obj_pair.second);

    /* Parallell collision loop for objects with cilkplus */
    std::map<int, std::vector<Model3D*>> recompute;
    cilk_for (auto i = 0; i < room_objects.size(); i++) {
        /* Abort program if object and room_object loop are same (must no happend) */
        assert(*object != *room_objects[i]);

        std::vector<Model3D*> collision_recompute = object->DetectCollision(room_objects[i], room_mutex_, proxy_parallell_);
        if (collision_recompute.size() > 0)
            recompute[room_objects[i]->id()] = collision_recompute;
    }

    /* end mutex lock */
    object->unlock();

    /* If collision detected, recompute older ones if exists */
    if (object->obstacle() != nullptr ) {
        try {
            for (auto & r : recompute.at(object->obstacle()->id())) {
                using engine::helpers::ProxyConfig;
                if (ProxyConfig::getSetting<int>("debug") >= ProxyConfig::kDEBUG_COLLISION)
                    std::cout << "Recompute " << object->id() << "::" << r->id() << std::endl;
                cilk_spawn PivotCollision(r);
            }
        }
        /* Silently catched out-of-range (not a problem) */
        catch (const std::out_of_range& oor) {}

        cilk_sync;
    }
}

}//namespace universe
}//namespace engine
