/**
 *  @file demo_game_factory.cc
 *  @brief Factory Class for demo game level
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/gameplay/demo_game_factory.h"

#include "nextfloor/gameplay/game_level.h"
#include "nextfloor/gameplay/game_loop.h"
#include "nextfloor/gameplay/game_timer.h"
#include "nextfloor/gameplay/head_camera.h"
#include "nextfloor/gameplay/player.h"

namespace nextfloor {

namespace gameplay {


DemoGameFactory::DemoGameFactory(HidFactory* hid_factory,
                                 ActionFactory* action_factory,
                                 RendererFactory* renderer_factory,
                                 nextfloor::objects::MeshFactory* mesh_factory)
{
    hid_factory_ = hid_factory;
    action_factory_ = action_factory;
    renderer_factory_ = renderer_factory;
    mesh_factory_ = mesh_factory;
}


std::unique_ptr<Loop> DemoGameFactory::MakeLoop() const
{
    auto level = MakeLevel();
    auto game_window = renderer_factory_->MakeSceneWindow();
    auto input_handler = hid_factory_->MakeInputHandler(*action_factory_, renderer_factory_);
    auto timer = MakeFrameTimer();

    return std::make_unique<GameLoop>(std::move(level), game_window, std::move(input_handler), std::move(timer));
}

std::unique_ptr<FrameTimer> DemoGameFactory::MakeFrameTimer() const
{
    return std::make_unique<GameTimer>();
}

std::unique_ptr<Level> DemoGameFactory::MakeLevel() const
{
    auto universe = GenerateUniverseWith3Rooms();
    auto player = MakePlayer(glm::vec3(0.0f, -2.0f, 5.0f));
    return std::make_unique<GameLevel>(std::move(universe), std::move(player), renderer_factory_);
}

std::unique_ptr<nextfloor::objects::Mesh> DemoGameFactory::MakePlayer(const glm::vec3& location) const
{
    auto border = mesh_factory_->MakeBorder(location, glm::vec3(0.4f));
    auto camera = MakeCamera(nullptr);
    return std::make_unique<Player>(location, std::move(border), std::move(camera));
}

std::unique_ptr<nextfloor::objects::Camera> DemoGameFactory::MakeCamera(nextfloor::objects::Mesh* owner) const
{
    return std::make_unique<HeadCamera>(owner, 3.14f, 0.0f);
}

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

std::unique_ptr<nextfloor::objects::Mesh> DemoGameFactory::GenerateUniverseWith3Rooms() const
{
    auto universe = mesh_factory_->MakeUniverse();

    auto first_room_location = glm::vec3(0.0f);
    auto room = mesh_factory_->MakeRoom(first_room_location);
    auto room_dimension = room->dimension();
    room->add_child(mesh_factory_->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
    room->add_child(mesh_factory_->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
    universe->add_child(std::move(room));

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = mesh_factory_->MakeRoom(room_location);
        room->add_child(mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f)));
        room->add_child(mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f)));
        universe->add_child(std::move(room));
    }

    universe->InitChildsIntoGrid();

    return universe;
}


}  // namespace gameplay

}  // namespace nextfloor