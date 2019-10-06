/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/gameplay/game_loop.h"
#include "nextfloor/core/common_services.h"

#include "nextfloor/hid/mouse_hid_factory.h"

int main(int argc, char* argv[])
{
    using nextfloor::core::CommonServices;

    /* Init Config */
    CommonServices::getConfig()->Initialize();

    /* Manage program parameters */
    CommonServices::getConfig()->ManageProgramParameters(argc, argv);

    /* Init GL Scene */
    nextfloor::hid::MouseHidFactory hid_factory;
    nextfloor::gameplay::GameLoop game_loop(hid_factory);

    /* Frame Loop */
    game_loop.Loop();

    CommonServices::getExit()->ExitOnSuccess();
}
