/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/hid/mouse_hid_factory.h"
#include "nextfloor/gameplay/game_loop.h"

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

    /* Init GL Scene */
    nextfloor::renderer::GlRendererFactory renderer_factory;
    nextfloor::objects::ModelMeshFactory game_factory;
    nextfloor::core::ServicesCoreFactory core_factory;
    nextfloor::hid::MouseHidFactory hid_factory;
    nextfloor::actions::SpriteActionFactory action_factory;
    nextfloor::gameplay::GameLoop game_loop(hid_factory, core_factory, game_factory, action_factory, &renderer_factory);

    /* Frame Loop */
    game_loop.Loop(&renderer_factory);

    CommonServices::getExit()->ExitOnSuccess();
}
