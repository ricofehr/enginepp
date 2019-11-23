/**
 *  @file demo_game_factory.cc
 *  @brief Factory Class for demo game level
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/gameplay/demo_game_factory.h"

#include "nextfloor/gameplay/game_level.h"
#include "nextfloor/gameplay/game_loop.h"
#include "nextfloor/gameplay/game_timer.h"

namespace nextfloor {

namespace gameplay {


DemoGameFactory::DemoGameFactory(HidFactory* hid_factory,
                                 RendererFactory* renderer_factory,
                                 nextfloor::playground::GroundFactory* ground_factory,
                                 nextfloor::thing::ThingFactory* thing_factory,
                                 nextfloor::character::CharacterFactory* character_factory,
                                 nextfloor::physic::CollisionEngineFactory* collision_engine_factory)
{
    hid_factory_ = hid_factory;
    renderer_factory_ = renderer_factory;
    ground_factory_ = ground_factory;
    thing_factory_ = thing_factory;
    character_factory_ = character_factory;
    collision_engine_factory_ = collision_engine_factory;
}

std::unique_ptr<Loop> DemoGameFactory::MakeLoop() const
{
    auto level = MakeLevel();
    auto game_window = renderer_factory_->GetOrMakeSceneWindow();
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
      std::move(universe), std::move(player), collision_engine_factory_->MakeCollisionEngine(), renderer_factory_);
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith3Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(0.0f);
    std::vector<glm::vec3> rocks_location = {glm::vec3(-3.0f, -1.5f, -4.0f), glm::vec3(3.0f, -2.5f, -5.5f)};
    std::vector<glm::vec3> rocks_movement = {glm::vec3(0.3, 0.01, 0.05), glm::vec3(0.3, 0.01, 0.05)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    auto room_dimension = rooms[0]->dimension();

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    return ground_factory_->MakeUniverse(std::move(rooms));
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith7Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(0.0f);
    std::vector<glm::vec3> rocks_location = {glm::vec3(-3.0f, -1.5f, -4.0f), glm::vec3(3.0f, -2.5f, -5.5f)};
    std::vector<glm::vec3> rocks_movement = {glm::vec3(0.3, 0.01, 0.05), glm::vec3(0.3, 0.01, 0.05)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    auto room_dimension = rooms[0]->dimension();

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = -1.0f; factor_z <= 1.0f; factor_z += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    factor_y = 0.0f;
    factor_z = 0.0f;
    for (factor_x = -1.0f; factor_x <= 1.0f; factor_x += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }


    factor_x = 0.0f;
    factor_z = 0.0f;
    for (factor_y = -1.0f; factor_y <= 1.0f; factor_y += 2.0f) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    return ground_factory_->MakeUniverse(std::move(rooms));
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith27Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(0.0f);
    std::vector<glm::vec3> rocks_location = {glm::vec3(-3.0f, -1.5f, -4.0f), glm::vec3(3.0f, -2.5f, -5.5f)};
    std::vector<glm::vec3> rocks_movement = {glm::vec3(0.3, 0.01, 0.05), glm::vec3(0.3, 0.01, 0.05)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    auto room_dimension = rooms[0]->dimension();

    /* create 26 more rooms around the first one */
    for (auto factor_x = -1.0f; factor_x <= 1.0f; factor_x++) {
        for (auto factor_y = -1.0f; factor_y <= 1.0f; factor_y++) {
            for (auto factor_z = -1.0f; factor_z <= 1.0f; factor_z++) {
                auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
                if (room_location != first_room_location) {
                    rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
                }
            }
        }
    }

    return ground_factory_->MakeUniverse(std::move(rooms));
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateRoom(glm::vec3 location,
                                                                             std::vector<glm::vec3> rocks_location,
                                                                             std::vector<glm::vec3> rocks_movement) const
{
    std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> room_objects;

    assert(rocks_location.size() == rocks_movement.size());

    for (int cnt = 0; cnt < rocks_location.size(); cnt++) {
        auto rock_location = location + rocks_location[cnt];
        auto rock_movement = rocks_movement[cnt];

        if (cnt % 2 == 0) {
            room_objects.push_back(thing_factory_->MakeRock(rock_location, rock_movement));
        }
        else {
            room_objects.push_back(thing_factory_->MakeLittleRock(rock_location, rock_movement));
        }
    }

    return ground_factory_->MakeRoom(location, std::move(room_objects));
}

}  // namespace gameplay

}  // namespace nextfloor