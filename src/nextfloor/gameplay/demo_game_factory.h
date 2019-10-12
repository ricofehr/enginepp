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
#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/camera.h"

#include "nextfloor/gameplay/hid_factory.h"
#include "nextfloor/gameplay/renderer_factory.h"
#include "nextfloor/objects/mesh_factory.h"
#include "nextfloor/objects/character_factory.h"
#include "nextfloor/objects/physic_factory.h"

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
                    nextfloor::objects::MeshFactory* mesh_factory,
                    nextfloor::objects::CharacterFactory* character_factory,
                    nextfloor::objects::PhysicFactory* physic_factory);
    ~DemoGameFactory() final = default;

    std::unique_ptr<Loop> MakeLoop() const override;
    std::unique_ptr<Level> MakeLevel() const override;

private:
    std::unique_ptr<FrameTimer> MakeFrameTimer() const;

    std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith3Rooms() const;
    std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith7Rooms() const;
    std::unique_ptr<nextfloor::objects::Mesh> GenerateUniverseWith27Rooms() const;

    HidFactory* hid_factory_{nullptr};
    RendererFactory* renderer_factory_{nullptr};
    nextfloor::objects::MeshFactory* mesh_factory_{nullptr};
    nextfloor::objects::CharacterFactory* character_factory_{nullptr};
    nextfloor::objects::PhysicFactory* physic_factory_{nullptr};
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_DEMOGAMEFACTORY_H_
