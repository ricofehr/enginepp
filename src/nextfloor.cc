/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/gameplay/game_loop.h"
#include "nextfloor/core/common_services.h"


int main(int argc, char* argv[])
{
    using nextfloor::core::CommonServices;
    using nextfloor::gameplay::GameLoop;

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
