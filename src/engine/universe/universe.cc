/*
* Universe class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/universe.h"

#include <glm/glm.hpp>
#include <GL/glew.h>

#include <iostream>
#include <memory>


namespace engine {
namespace universe {

/* Constructor */
Universe::Universe()
{
    InitCamera();
    InitRooms();
}

/* Create Rooms for gl scene */
void Universe::InitRooms()
{
    /* Locations for 4 rooms */
    std::vector<glm::vec4> locations = {
        glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(30.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 30.0f, 1.0f),
        glm::vec4(30.0f, 0.0f, 30.0f, 1.0f)
    };

    /* Choose wich face in room have Door */
    std::vector<std::vector<bool>> is_doors = {
        {true, true, false, false},
        {true, false, false, true},
        {false, true, true, false},
        {false, false, true, true},
    };

    std::cout << "Init Rooms\n";

    proxy_cl_ = std::make_unique<engine::helpers::ProxyCL>();
    proxy_cl_->InitCollisionCL();

    /* Add rooms on current universe object */
    std::vector<std::vector<bool>>::iterator it_doors = is_doors.begin();
    for (auto &loc : locations) {
        auto room_ptr{std::make_unique<Room>(loc, *it_doors, proxy_cl_.get())};
        rooms_.push_back(std::move(room_ptr));
        it_doors++;
    }

    active_room_ = nullptr;
}

/* Create Camera for gl scene */
void Universe::InitCamera()
{
    std::cout << "Init Camera\n";
    cam_ = std::make_unique<Camera>(0.0f, 1.0f, 5.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
}

/* Compute and draw next hop for the gl scene */
void Universe::NextHop()
{
    if (active_room_ == nullptr
        || !cam_->IsInRoom(*active_room_)) {

        for (auto &r : rooms_) {
            if (cam_->IsInRoom(*r)) {
                active_room_ = r.get();
                break;
            }
        }
    }

    active_room_->DetectCollision();
    cam_->Draw();
    active_room_->Draw();
}

/* Destructor */
Universe::~Universe() {}

}//namespace universe
}//namespace engine
