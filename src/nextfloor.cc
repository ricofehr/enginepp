/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>

#include "nextfloor/universe/random_universe_factory.h"
#include "nextfloor/core/config_engine.h"
#include "nextfloor/renderer/loopgl.h"

int main(int argc, char* argv[])
{
    using nextfloor::universe::Universe;
    using nextfloor::universe::RandomUniverseFactory;
    using nextfloor::renderer::LoopGL;
    using nextfloor::core::ConfigEngine;

    /* Reset seed */
    srand (time(NULL));

    /* Init Config */
    ConfigEngine::InitConfig();

    /* Manage program parameters */
    ConfigEngine::ManageProgramParameters(argc, argv);

	/* Init world */
    RandomUniverseFactory factory;
    LoopGL::Instance()->InitGL();
    factory.GenerateBuffers();
    std::unique_ptr<Universe> universe{factory.GenerateUniverse()};

    /* Launch GL Scene */
    LoopGL::Instance()->Loop(universe.get());
}
