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
                                 RendererFactory* renderer_factory,
                                 nextfloor::objects::MeshFactory* mesh_factory,
                                 nextfloor::objects::CharacterFactory* character_factory,
                                 nextfloor::objects::PhysicFactory* physic_factory)
{
    hid_factory_ = hid_factory;
    renderer_factory_ = renderer_factory;
    mesh_factory_ = mesh_factory;
    character_factory_ = character_factory;
    physic_factory_ = physic_factory;
}

std::unique_ptr<Loop> DemoGameFactory::MakeLoop() const
{
    auto level = MakeLevel();
    auto game_window = renderer_factory_->MakeSceneWindow();
    auto input_handler = hid_factory_->MakeInputHandler();
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
    auto player = character_factory_->MakePlayer(glm::vec3(0.0f, -2.0f, 5.0f));
    return std::make_unique<GameLevel>(
      std::move(universe), std::move(player), physic_factory_->MakeCollisionEngine(), renderer_factory_);
}

std::unique_ptr<nextfloor::objects::Mesh> DemoGameFactory::GenerateUniverseWith3Rooms() const
{
    auto universe = mesh_factory_->MakeUniverse();

    auto rock_movement = glm::vec3(0.3, 0.01, 0.05);
    auto first_room_location = glm::vec3(0.0f);
    auto room = mesh_factory_->MakeRoom(first_room_location);
    auto room_dimension = room->dimension();
    room->add_child(mesh_factory_->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
    room->add_child(mesh_factory_->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
    universe->add_child(std::move(room));

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = mesh_factory_->MakeRoom(room_location);
        room->add_child(mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
        room->add_child(mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
        universe->add_child(std::move(room));
    }

    universe->InitChildsIntoGrid();

    return universe;
}

std::unique_ptr<nextfloor::objects::Mesh> DemoGameFactory::GenerateUniverseWith7Rooms() const
{
    auto universe = mesh_factory_->MakeUniverse();

    auto rock_movement = glm::vec3(0.3, 0.01, 0.05);
    auto first_room_location = glm::vec3(0.0f);
    auto room = mesh_factory_->MakeRoom(first_room_location);
    auto room_dimension = room->dimension();
    room->add_child(mesh_factory_->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
    room->add_child(mesh_factory_->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
    universe->add_child(std::move(room));

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = mesh_factory_->MakeRoom(room_location);
        room->add_child(mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
        room->add_child(mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
        universe->add_child(std::move(room));
    }

    factor_y = 0.0f;
    factor_z = 0.0f;
    for (factor_x = -1.0f; factor_x <= 1.0f; factor_x += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = mesh_factory_->MakeRoom(room_location);
        room->add_child(mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
        room->add_child(mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
        universe->add_child(std::move(room));
    }


    factor_x = 0.0f;
    factor_z = 0.0f;
    for (factor_y = -1.0f; factor_y <= 1.0f; factor_y += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        auto room = mesh_factory_->MakeRoom(room_location);
        room->add_child(mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
        room->add_child(mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
        universe->add_child(std::move(room));
    }

    universe->InitChildsIntoGrid();

    return universe;
}

std::unique_ptr<nextfloor::objects::Mesh> DemoGameFactory::GenerateUniverseWith27Rooms() const
{
    auto universe = mesh_factory_->MakeUniverse();

    auto rock_movement = glm::vec3(0.3, 0.01, 0.05);
    auto first_room_location = glm::vec3(0.0f);
    auto first_room = mesh_factory_->MakeRoom(first_room_location);
    auto room_dimension = first_room->dimension();
    first_room->add_child(mesh_factory_->MakeRock(first_room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
    first_room->add_child(
      mesh_factory_->MakeLittleRock(first_room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
    universe->add_child(std::move(first_room));

    /* create 26 more rooms around the first one */
    for (auto factor_x = -1.0f; factor_x <= 1.0f; factor_x++) {
        for (auto factor_y = -1.0f; factor_y <= 1.0f; factor_y++) {
            for (auto factor_z = -1.0f; factor_z <= 1.0f; factor_z++) {
                auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
                if (room_location != first_room_location) {
                    auto room = mesh_factory_->MakeRoom(room_location);
                    room->add_child(
                      mesh_factory_->MakeRock(room_location + glm::vec3(-3.0f, -1.5f, -4.0f), rock_movement));
                    room->add_child(
                      mesh_factory_->MakeLittleRock(room_location + glm::vec3(3.0f, -2.5f, -5.5f), rock_movement));
                    universe->add_child(std::move(room));
                }
            }
        }
    }

    universe->InitChildsIntoGrid();
    return universe;
}


}  // namespace gameplay

}  // namespace nextfloor