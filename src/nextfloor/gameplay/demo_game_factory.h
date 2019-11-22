/**
 *  @file demo_game_factory.h
 *  @brief Concrete Factory Class Implementation for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_

#include "nextfloor/gameplay/game_factory.h"

#include "nextfloor/gameplay/loop.h"
#include "nextfloor/gameplay/level.h"
#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/mesh/mesh.h"

#include "nextfloor/gameplay/hid_factory.h"
#include "nextfloor/gameplay/renderer_factory.h"
#include "nextfloor/playground/ground_factory.h"
#include "nextfloor/thing/thing_factory.h"
#include "nextfloor/character/character_factory.h"
#include "nextfloor/mesh/physic_factory.h"

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
                    nextfloor::playground::GroundFactory* ground_factory,
                    nextfloor::thing::ThingFactory* thing_factory,
                    nextfloor::character::CharacterFactory* character_factory,
                    nextfloor::mesh::PhysicFactory* physic_factory);
    ~DemoGameFactory() final = default;

    std::unique_ptr<Loop> MakeLoop() const override;
    std::unique_ptr<Level> MakeLevel() const override;

private:
    std::unique_ptr<FrameTimer> MakeFrameTimer() const;

    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith3Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith7Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateUniverseWith27Rooms() const;
    std::unique_ptr<nextfloor::playground::Ground> GenerateRoom(glm::vec3 location,
                                                                std::vector<glm::vec3> rocks_location,
                                                                std::vector<glm::vec3> rocks_movement) const;

    HidFactory* hid_factory_{nullptr};
    RendererFactory* renderer_factory_{nullptr};
    nextfloor::playground::GroundFactory* ground_factory_{nullptr};
    nextfloor::thing::ThingFactory* thing_factory_{nullptr};
    nextfloor::character::CharacterFactory* character_factory_{nullptr};
    nextfloor::mesh::PhysicFactory* physic_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_
