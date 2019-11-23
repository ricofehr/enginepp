/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/hid/mouse_hid_factory.h"
#include "nextfloor/gameplay/demo_game_factory.h"
#include "nextfloor/character/game_character_factory.h"
#include "nextfloor/camera/game_camera_factory.h"
#include "nextfloor/physic/mesh_border_factory.h"
#include "nextfloor/physic/game_collision_engine_factory.h"
#include "nextfloor/polygon/mesh_polygon_factory.h"
#include "nextfloor/layout/mesh_grid_factory.h"

#include "nextfloor/renderer/gl_renderer_factory.h"
#include "nextfloor/playground/game_ground_factory.h"
#include "nextfloor/thing/game_thing_factory.h"
#include "nextfloor/core/services_core_factory.h"
#include "nextfloor/action/sprite_action_factory.h"
#include "nextfloor/core/common_services.h"

int main(int argc, char* argv[])
{
    using nextfloor::core::CommonServices;

    /* Init Config */
    CommonServices::getConfig()->Initialize();

    /* Manage program parameters */
    CommonServices::getConfig()->ManageProgramParameters(argc, argv);

    /* Init Factories */
    nextfloor::polygon::MeshPolygonFactory polygon_factory;
    nextfloor::action::SpriteActionFactory action_factory;
    nextfloor::renderer::GlRendererFactory renderer_factory;
    nextfloor::layout::MeshGridFactory grid_factory;
    nextfloor::physic::MeshBorderFactory border_factory;
    nextfloor::physic::GameCollisionEngineFactory collision_engine_factory;
    nextfloor::camera::GameCameraFactory camera_factory;
    nextfloor::character::GameCharacterFactory character_factory(&camera_factory, &border_factory);
    nextfloor::thing::GameThingFactory thing_factory(&polygon_factory, &border_factory);
    nextfloor::playground::GameGroundFactory ground_factory(&thing_factory, &grid_factory, &border_factory);
    nextfloor::hid::MouseHidFactory hid_factory(&action_factory, &renderer_factory);
    nextfloor::gameplay::DemoGameFactory game_factory(
      &hid_factory, &renderer_factory, &ground_factory, &thing_factory, &character_factory, &collision_engine_factory);

    auto game_loop = game_factory.MakeLoop();

    /* Frame Loop */
    game_loop->RunLoop();

    CommonServices::getExit()->ExitOnSuccess();
}
