/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>
#include <tbb/tbb.h>

#include "nextfloor/universe/factory/random_universe_factory.h"
#include "nextfloor/core/common_services.h"
#include "nextfloor/job/game_loop.h"
#include "nextfloor/renderer/game_window.h"

int main(int argc, char* argv[])
{
    using nextfloor::universe::Universe;
    using nextfloor::universe::dynamic::Camera;
    using nextfloor::universe::factory::RandomUniverseFactory;
    using nextfloor::job::GameLoop;
    using nextfloor::renderer::GameWindow;
    using nextfloor::core::CommonServices;

    /* Init Config */
    CommonServices::getConfig().Initialize();

    /* Manage program parameters */
    CommonServices::getConfig().ManageProgramParameters(argc, argv);

    /* Manage Threads Parallelism : disable tbb parallelism if serial option, or set arbitrary thread number (default is to let tbb core decide) */
    std::unique_ptr<tbb::task_scheduler_init> tbb_threads_config{nullptr};
    if (CommonServices::getConfig().getSetting<int>("workers_count")) {
        tbb_threads_config = std::make_unique<tbb::task_scheduler_init>(CommonServices::getConfig().getSetting<int>("workers_count"));
    }

	/* Init world */
    RandomUniverseFactory factory;
    GameWindow game_window;
    GameLoop game_loop(&game_window);
    game_window.Initialization();
    factory.GenerateBuffers();
    std::unique_ptr<Universe> universe{factory.GenerateUniverse()};

    /* Launch GL Scene */
    game_window.SetCamera((Camera*)universe->get_camera());
    game_loop.Loop(universe.get());
}
