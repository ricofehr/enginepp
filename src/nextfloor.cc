/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>
#include <tbb/tbb.h>

#include "nextfloor/universe/factory/random_universe_factory.h"
#include "nextfloor/core/config_engine.h"
#include "nextfloor/job/game_loop.h"

int main(int argc, char* argv[])
{
    using nextfloor::universe::Universe;
    using nextfloor::universe::factory::RandomUniverseFactory;
    using nextfloor::job::GameLoop;
    using nextfloor::core::ConfigEngine;

    /* Reset seed */
    srand(time(NULL));

    /* Init Config */
    ConfigEngine::InitConfig();

    /* Manage program parameters */
    ConfigEngine::ManageProgramParameters(argc, argv);

    /* Manage Threads Parallelism : disable tbb parallelism if serial option, or set arbitrary thread number (default is to let tbb core decide) */
    std::unique_ptr<tbb::task_scheduler_init> tbb_threads_config{nullptr};
    if (ConfigEngine::getSetting<int>("workers_count")) {
        tbb_threads_config = std::make_unique<tbb::task_scheduler_init>(ConfigEngine::getSetting<int>("workers_count"));
    }

	/* Init world */
    RandomUniverseFactory factory;
    GameLoop::Instance()->InitGL();
    factory.GenerateBuffers();
    std::unique_ptr<Universe> universe{factory.GenerateUniverse()};

    /* Launch GL Scene */
    GameLoop::Instance()->Loop(universe.get());
}
