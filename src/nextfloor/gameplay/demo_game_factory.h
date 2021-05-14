/**
 *  @file demo_game_factory.h
 *  @brief Concrete Factory Class Implementation for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_

#include "nextfloor/gameplay/game_factory.h"

#include <memory>
#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/playground/ground.h"

#include "nextfloor/gameplay/hid_factory.h"
#include "nextfloor/gameplay/renderer_factory.h"
#include "nextfloor/gameplay/menu_factory.h"
#include "nextfloor/playground/ground_factory.h"
#include "nextfloor/thing/thing_factory.h"
#include "nextfloor/character/character_factory.h"
#include "nextfloor/physic/collision_engine_factory.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class DemoGameFactory
 *  @brief Concrete Factory for a demo gameplay system
 */
class DemoGameFactory : public GameFactory {

public:
    DemoGameFactory(HidFactory* hid_factory,
                    RendererFactory* renderer_factory,
                    MenuFactory* menu_factory,
                    nextfloor::playground::GroundFactory* ground_factory,
                    nextfloor::thing::ThingFactory* thing_factory,
                    nextfloor::character::CharacterFactory* character_factory,
                    nextfloor::physic::CollisionEngineFactory* collision_engine_factory);
    ~DemoGameFactory() final = default;

    std::unique_ptr<Loop> MakeLoop() const override;
    std::unique_ptr<Level> MakeLevel() const override;

private:
    static constexpr float kPlayerLocationX = 0.0f;
    static constexpr float kPlayerLocationY = -5.0f;
    static constexpr float kPlayerLocationZ = 5.0f;

    static constexpr float kFirstRoomLocationX = 0.0f;
    static constexpr float kFirstRoomLocationY = 0.0f;
    static constexpr float kFirstRoomLocationZ = 0.0f;

    static constexpr float kRock1LocationX = -3.0f;
    static constexpr float kRock1LocationY = -1.5f;
    static constexpr float kRock1LocationZ = -4.0f;
    static constexpr float kRock1MoveX = 0.3f;
    static constexpr float kRock1MoveY = 0.01f;
    static constexpr float kRock1MoveZ = 0.05f;

    static constexpr float kRock2LocationX = 3.0f;
    static constexpr float kRock2LocationY = -2.5f;
    static constexpr float kRock2LocationZ = -5.5f;
    static constexpr float kRock2MoveX = 0.3f;
    static constexpr float kRock2MoveY = 0.01f;
    static constexpr float kRock2MoveZ = 0.05f;

    static constexpr float kRock3LocationX = 1.0f;
    static constexpr float kRock3LocationY = -4.5f;
    static constexpr float kRock3LocationZ = -5.5f;
    static constexpr float kRock3MoveX = 0.0f;
    static constexpr float kRock3MoveY = 0.0f;
    static constexpr float kRock3MoveZ = 0.0f;

    static constexpr float kRock4LocationX = -1.0f;
    static constexpr float kRock4LocationY = -6.0f;
    static constexpr float kRock4LocationZ = -4.5f;
    static constexpr float kRock4MoveX = 0.0f;
    static constexpr float kRock4MoveY = 0.0f;
    static constexpr float kRock4MoveZ = 0.0f;


    static constexpr float kFactor0 = -1.0f;
    static constexpr float kFactor1 = 1.0f;
    static constexpr float kFactorDelta = 2.0f;

    std::unique_ptr<FrameTimer> MakeFrameTimer() const;

    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith3Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith7Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith27Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateRoom(glm::vec3 location,
                                                                std::vector<glm::vec3> rocks_location,
                                                                std::vector<glm::vec3> rocks_movement) const;

    HidFactory* hid_factory_{nullptr};
    RendererFactory* renderer_factory_{nullptr};
    MenuFactory* menu_factory_{nullptr};
    nextfloor::playground::GroundFactory* ground_factory_{nullptr};
    nextfloor::thing::ThingFactory* thing_factory_{nullptr};
    nextfloor::character::CharacterFactory* character_factory_{nullptr};
    nextfloor::physic::CollisionEngineFactory* collision_engine_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_
