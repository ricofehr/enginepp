/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/hid/mouse_hid_factory.h"
#include "nextfloor/gameplay/demo_game_factory.h"
#include "nextfloor/gameplay/game_character_factory.h"
#include "nextfloor/physics/mesh_physic_factory.h"
#include "nextfloor/polygons/mesh_polygon_factory.h"
#include "nextfloor/grid/mesh_grid_factory.h"

#include "nextfloor/renderer/gl_renderer_factory.h"
#include "nextfloor/objects/model_mesh_factory.h"
#include "nextfloor/core/services_core_factory.h"
#include "nextfloor/actions/sprite_action_factory.h"
#include "nextfloor/core/common_services.h"

int main(int argc, char* argv[])
{
    using nextfloor::core::CommonServices;

    /* Init Config */
    CommonServices::getConfig()->Initialize();

    /* Manage program parameters */
    CommonServices::getConfig()->ManageProgramParameters(argc, argv);

    /* Init Factories */
    nextfloor::polygons::MeshPolygonFactory polygon_factory;
    nextfloor::actions::SpriteActionFactory action_factory;
    nextfloor::renderer::GlRendererFactory renderer_factory;
    nextfloor::grid::MeshGridFactory grid_factory;
    nextfloor::physics::MeshPhysicFactory physic_factory(&polygon_factory);
    nextfloor::gameplay::GameCharacterFactory character_factory(&physic_factory);
    nextfloor::objects::ModelMeshFactory mesh_factory(&polygon_factory, &grid_factory, &physic_factory);
    nextfloor::hid::MouseHidFactory hid_factory(&action_factory, &renderer_factory);
    nextfloor::gameplay::DemoGameFactory game_factory(
      &hid_factory, &renderer_factory, &mesh_factory, &character_factory, &physic_factory);

    auto game_loop = game_factory.MakeLoop();

    /* Frame Loop */
    game_loop->RunLoop();

    CommonServices::getExit()->ExitOnSuccess();
}
