/*
* Universe class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/universe.h"

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <memory>
#include <cilk/cilk.h>

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
    InitCamera();
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
    /* Check objects count into config file */
    using engine::helpers::ProxyConfig;
    nbrooms_ = ProxyConfig::getSetting<int>("rooms_count");

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

    if (ProxyConfig::getSetting<int>("debug") > 0)
        std::cout << "Init Rooms\n";

    /* Add rooms on current universe object */
    std::vector<std::vector<bool>>::iterator it_doors = is_doors.begin();
    for (auto &loc : locations) {
        auto room_ptr{std::make_unique<Room>(loc, *it_doors, cam_.get(), proxy_parallell_.get())};
        rooms_.push_back(std::move(room_ptr));
        it_doors++;
        if (rooms_.size() == nbrooms_)
            break;
    }

    active_room_ = nullptr;
}

/* Create Camera for gl scene */
void Universe::InitCamera()
{
    using engine::helpers::ProxyConfig;
    if (ProxyConfig::getSetting<int>("debug") > 0)
        std::cout << "Init Camera\n";

    cam_ = std::make_unique<Camera>(0.0f, 1.0f, 5.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f);
}

/* Compute and draw next hop for the gl scene */
void Universe::NextHop()
{
    using engine::helpers::ProxyConfig;

    if (active_room_ == nullptr
        || !cam_->IsInRoom(*active_room_)) {

        for (auto &r : rooms_) {
            if (cam_->IsInRoom(*r)) {
                active_room_ = r.get();
                break;
            }
        }
    }

    /* Record moving orders for camera */
    cam_->Move();
    /* Detect collision in rooms */
    cilk_for (auto cnt = 0; cnt < rooms_.size(); cnt++) {
        rooms_[cnt]->DetectCollision();
    }

    cam_->PrepareDraw(cam_.get());
    active_room_->Draw();

    double current_time = glfwGetTime();
    cilk_for (auto cnt = 0; cnt < rooms_.size(); cnt++) {
        rooms_[cnt]->ReinitGrid();
    }

    /* GL functions during object generate, then needs serial execution */
    if (ProxyConfig::getSetting<bool>("load_objects_seq")) {
        for (auto &r : rooms_) {
            if (!r->IsFull() && current_time - kLastTime >= 0.5f)
                r->GenerateRandomObject();
        }

        if (current_time - kLastTime >= 0.5f)
            kLastTime = current_time;
    }
}

}//namespace universe
}//namespace engine
