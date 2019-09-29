/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/controller/game_loop.h"
#include "nextfloor/core/common_services.h"


int main(int argc, char* argv[])
{
    using nextfloor::controller::GameLoop;
    using nextfloor::core::CommonServices;

    /* Init Config */
    CommonServices::getConfig()->Initialize();

    /* Manage program parameters */
    CommonServices::getConfig()->ManageProgramParameters(argc, argv);

    /* Init GL Scene */
    GameLoop game_loop;

    /* Frame Loop */
    game_loop.Loop();

    CommonServices::getExit()->ExitOnSuccess();
}
