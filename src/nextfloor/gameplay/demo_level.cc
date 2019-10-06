/**
 *  @file demo_level.cc
 *  @brief DemoLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/demo_level.h"

#include <tbb/tbb.h>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

namespace {

// std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith27Rooms()
// {
//     auto factory = nextfloor::core::CommonServices::getFactory();
//     auto universe = factory->MakeUniverse();

//     auto first_room_location = glm::vec3(0.0f);
//     auto first_room = factory->MakeRoom(first_room_location);
//     auto room_dimension = first_room->dimension();
//     first_room->add_child(factory->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//     first_room->add_child(factory->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//     universe->add_child(std::move(first_room));

//     /* create 26 more rooms around the first one */
//     for (auto factor_x = -1.0f; factor_x <= 1.0f; factor_x++) {
//         for (auto factor_y = -1.0f; factor_y <= 1.0f; factor_y++) {
//             for (auto factor_z = -1.0f; factor_z <= 1.0f; factor_z++) {
//                 auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
//                 if (room_location != first_room_location) {
//                     auto room = factory->MakeRoom(room_location);
//                     room->add_child(factory->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//                     room->add_child(factory->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//                     universe->add_child(std::move(room));
//                 }
//             }
//         }
//     }

//     universe->InitChildsIntoGrid();
//     return universe;
// }


// std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith7Rooms()
// {
//     auto factory = nextfloor::core::CommonServices::getFactory();
//     auto universe = factory->MakeUniverse();

//     auto first_room_location = glm::vec3(0.0f);
//     auto room = factory->MakeRoom(first_room_location);
//     auto room_dimension = room->dimension();
//     room->add_child(factory->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//     room->add_child(factory->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//     universe->add_child(std::move(room));

//     auto factor_x = 0.0f;
//     auto factor_y = 0.0f;
//     float factor_z;
//     for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
//         auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
//         auto room = factory->MakeRoom(room_location);
//         room->add_child(factory->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//         room->add_child(factory->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//         universe->add_child(std::move(room));
//     }

//     factor_y = 0.0f;
//     factor_z = 0.0f;
//     for (factor_x = -1.0f; factor_x <= 1.0f; factor_x += 2.0f) {
//         auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
//         auto room = factory->MakeRoom(room_location);
//         room->add_child(factory->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//         room->add_child(factory->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//         universe->add_child(std::move(room));
//     }


//     factor_x = 0.0f;
//     factor_z = 0.0f;
//     for (factor_y = -1.0f; factor_y <= 1.0f; factor_y += 2.0f) {
//         auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
//         auto room = factory->MakeRoom(room_location);
//         room->add_child(factory->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
//         room->add_child(factory->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
//         universe->add_child(std::move(room));
//     }

//     universe->InitChildsIntoGrid();

//     return universe;
// }

std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith3Rooms(const nextfloor::objects::MeshFactory& factory)
{
    auto universe = factory.MakeUniverse();

    auto first_room_location = glm::vec3(0.0f);
    auto room = factory.MakeRoom(first_room_location);
    auto room_dimension = room->dimension();
    room->add_child(factory.MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
    room->add_child(factory.MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
    universe->add_child(std::move(room));

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = factory.MakeRoom(room_location);
        room->add_child(factory.MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
        room->add_child(factory.MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
        universe->add_child(std::move(room));
    }

    universe->InitChildsIntoGrid();

    return universe;
}


}  // namespace

DemoLevel::DemoLevel(const nextfloor::objects::MeshFactory& factory)
{
    auto player = factory.MakePlayer(glm::vec3(0.0f, -2.0f, 5.0f));
    player_ = player.get();
    GenerateUniverse(factory);
    universe_->AddIntoChild(std::move(player));
    game_cameras_ = universe_->all_cameras();
    SetActiveCamera(player_->camera());
}

void DemoLevel::SetActiveCamera(nextfloor::objects::Camera* active_camera)
{
    std::list<nextfloor::objects::Camera*>::iterator it;
    for (it = game_cameras_.begin(); it != game_cameras_.end(); ++it) {
        if (*it == active_camera) {
            game_cameras_.remove(active_camera);
            game_cameras_.push_front(active_camera);
            break;
        }
    }
}


void DemoLevel::GenerateUniverse(const nextfloor::objects::MeshFactory& factory)
{
    universe_ = GenerateUniverseWith3Rooms(factory);
}

void DemoLevel::UpdateCameraOrientation(HIDPointer angles, float input_fov)
{
    auto active_camera = game_cameras_.front();
    active_camera->ComputeFOV(input_fov);
    active_camera->increment_angles(angles.horizontal_delta_angle, angles.vertical_delta_angle);
    active_camera->ComputeOrientation();
}

void DemoLevel::ExecutePlayerAction(Action* command, double elapsed_time)
{
    command->execute(player_, elapsed_time);
}


void DemoLevel::Move()
{
    universe_->DetectCollision();
    universe_->Move();
}

void DemoLevel::Draw()
{
    auto active_camera = game_cameras_.front();
    universe_->PrepareDraw(*active_camera);
    auto polygons = universe_->GetPolygonsReadyToDraw(*active_camera);
    auto factory = nextfloor::core::CommonServices::getFactory();
    for (const auto& polygon : polygons) {
        auto renderer = factory->MakeCubeRenderer(polygon->texture());
        renderer->Draw(polygon->mvp());
    }
}

}  // namespace gameplay

}  // namespace nextfloor
