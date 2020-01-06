/**
 *  @file demo_game_factory.cc
 *  @brief Factory Class for demo game level
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/gameplay/demo_game_factory.h"

#include <utility>

#include "nextfloor/gameplay/scene_window.h"
#include "nextfloor/gameplay/input_handler.h"
#include "nextfloor/character/character.h"
#include "nextfloor/physic/collision_engine.h"
#include "nextfloor/playground/ground.h"

#include "nextfloor/gameplay/game_level.h"
#include "nextfloor/gameplay/game_loop.h"
#include "nextfloor/gameplay/game_timer.h"

#include "nextfloor/core/common_services.h"

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
    auto timer = MakeFrameTimer();
    auto level = MakeLevel();
    SceneWindow* game_window = renderer_factory_->GetOrMakeSceneWindow();
    std::unique_ptr<InputHandler> input_handler = hid_factory_->MakeInputHandler();

    return std::make_unique<GameLoop>(std::move(level), game_window, std::move(input_handler), std::move(timer));
}

std::unique_ptr<FrameTimer> DemoGameFactory::MakeFrameTimer() const
{
    return std::make_unique<GameTimer>();
}

std::unique_ptr<Level> DemoGameFactory::MakeLevel() const
{
    using nextfloor::playground::Ground;
    std::unique_ptr<Ground> universe = GenerateUniverseWith3Rooms();
    using nextfloor::character::Character;
    std::unique_ptr<Character> player
      = character_factory_->MakePlayer(glm::vec3(kPlayerLocationX, kPlayerLocationY, kPlayerLocationZ));

    using nextfloor::core::CommonServices;
    int granularity = CommonServices::getConfig()->getCollisionGranularity();
    int type = CommonServices::getConfig()->getParallellAlgoType();

    using nextfloor::physic::CollisionEngine;
    std::unique_ptr<CollisionEngine> collision_engine = collision_engine_factory_->MakeCollisionEngine(type, granularity);

    return std::make_unique<GameLevel>(
      std::move(universe), std::move(player), std::move(collision_engine), renderer_factory_);
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith3Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(kFirstRoomLocationX, kFirstRoomLocationY, kFirstRoomLocationZ);
    std::vector<glm::vec3> rocks_location = {glm::vec3(kRock1LocationX, kRock1LocationY, kRock1LocationZ),
                                             glm::vec3(kRock2LocationX, kRock2LocationY, kRock2LocationZ)};
    std::vector<glm::vec3> rocks_movement
      = {glm::vec3(kRock1MoveX, kRock1MoveY, kRock1MoveZ), glm::vec3(kRock2MoveX, kRock2MoveY, kRock2MoveZ)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    glm::vec3 room_dimension = rooms[0]->dimension();

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = kFactor0; factor_z <= kFactor1; factor_z += kFactorDelta) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    return ground_factory_->MakeUniverse(std::move(rooms));
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith7Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(kFirstRoomLocationX, kFirstRoomLocationY, kFirstRoomLocationZ);
    std::vector<glm::vec3> rocks_location = {glm::vec3(kRock1LocationX, kRock1LocationY, kRock1LocationZ),
                                             glm::vec3(kRock2LocationX, kRock2LocationY, kRock2LocationZ)};
    std::vector<glm::vec3> rocks_movement
      = {glm::vec3(kRock1MoveX, kRock1MoveY, kRock1MoveZ), glm::vec3(kRock2MoveX, kRock2MoveY, kRock2MoveZ)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    glm::vec3 room_dimension = rooms[0]->dimension();

    auto factor_x = 0.0f;
    auto factor_y = 0.0f;
    float factor_z;
    for (factor_z = kFactor0; factor_z <= kFactor1; factor_z += kFactorDelta) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    factor_y = 0.0f;
    factor_z = 0.0f;
    for (factor_x = kFactor0; factor_x <= kFactor1; factor_x += kFactorDelta) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }


    factor_x = 0.0f;
    factor_z = 0.0f;
    for (factor_y = kFactor0; factor_y <= kFactor1; factor_y += kFactorDelta) {
        auto room_location = first_room_location + glm::vec3(factor_x, factor_y, factor_z) * room_dimension;
        rooms.push_back(GenerateRoom(room_location, rocks_location, rocks_movement));
    }

    return ground_factory_->MakeUniverse(std::move(rooms));
}

std::unique_ptr<nextfloor::playground::Ground> DemoGameFactory::GenerateUniverseWith27Rooms() const
{
    std::vector<std::unique_ptr<nextfloor::playground::Ground>> rooms;

    auto first_room_location = glm::vec3(kFirstRoomLocationX, kFirstRoomLocationY, kFirstRoomLocationZ);
    std::vector<glm::vec3> rocks_location = {glm::vec3(kRock1LocationX, kRock1LocationY, kRock1LocationZ),
                                             glm::vec3(kRock2LocationX, kRock2LocationY, kRock2LocationZ)};
    std::vector<glm::vec3> rocks_movement
      = {glm::vec3(kRock1MoveX, kRock1MoveY, kRock1MoveZ), glm::vec3(kRock2MoveX, kRock2MoveY, kRock2MoveZ)};
    rooms.push_back(GenerateRoom(first_room_location, rocks_location, rocks_movement));
    glm::vec3 room_dimension = rooms[0]->dimension();

    /* create 26 more rooms around the first one */
    for (auto factor_x = kFactor0; factor_x <= kFactor1; factor_x++) {
        for (auto factor_y = kFactor0; factor_y <= kFactor1; factor_y++) {
            for (auto factor_z = kFactor0; factor_z <= kFactor1; factor_z++) {
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