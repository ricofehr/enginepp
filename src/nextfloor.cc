/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>
#include <tbb/tbb.h>

#include "nextfloor/universe/factory/demo_universe_factory.h"
#include "nextfloor/universe/factory/random_universe_factory.h"
#include "nextfloor/job/game_loop.h"
#include "nextfloor/renderer/game_window.h"

#include "nextfloor/core/common_services.h"
#include "nextfloor/core/terminal_log.h"
#include "nextfloor/core/frame_timer.h"
#include "nextfloor/core/config_file.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/generic_file_io.h"
#include "nextfloor/core/standard_random_generator.h"

int main(int argc, char* argv[])
{
    using nextfloor::universe::Universe;
    using nextfloor::universe::dynamic::Camera;
    using nextfloor::universe::factory::UniverseFactory;
    using nextfloor::universe::factory::RandomUniverseFactory;
    using nextfloor::universe::factory::DemoUniverseFactory;
    using nextfloor::job::GameLoop;
    using nextfloor::renderer::GameWindow;
    using nextfloor::core::CommonServices;
    using nextfloor::core::TerminalLog;
    using nextfloor::core::FrameTimer;
    using nextfloor::core::ConfigFile;
    using nextfloor::core::ProgramExit;
    using nextfloor::core::GenericFileIO;
    using nextfloor::core::StandardRandomGenerator;

    /* Init CommonServices */
    std::unique_ptr<TerminalLog> terminal_log = std::make_unique<TerminalLog>();
    CommonServices::provideLog(terminal_log.get());
    std::unique_ptr<FrameTimer> frame_timer  = std::make_unique<FrameTimer>();
    CommonServices::provideTimer(frame_timer.get());
    std::unique_ptr<ConfigFile> file_config = std::make_unique<ConfigFile>();
    CommonServices::provideConfig(file_config.get());
    std::unique_ptr<ProgramExit> program_exit = std::make_unique<ProgramExit>();
    CommonServices::provideExit(program_exit.get());
    std::unique_ptr<GenericFileIO> generic_file_io = std::make_unique<GenericFileIO>();
    CommonServices::provideFileIO(generic_file_io.get());
    std::unique_ptr<StandardRandomGenerator> standard_random_generator = std::make_unique<StandardRandomGenerator>();
    CommonServices::provideRandomGenerator(standard_random_generator.get());

    /* Init Config */
    CommonServices::getConfig()->Initialize();

    /* Manage program parameters */
    CommonServices::getConfig()->ManageProgramParameters(argc, argv);

    /* Manage Threads Parallelism */
    std::unique_ptr<tbb::task_scheduler_init> tbb_threads_config{nullptr};
    if (CommonServices::getConfig()->getThreadsCount()) {
        tbb_threads_config = std::make_unique<tbb::task_scheduler_init>(CommonServices::getConfig()->getThreadsCount());
    }

	/* Init world */
    std::unique_ptr<UniverseFactory> factory{nullptr};
    if (CommonServices::getConfig()->getUniverseFactoryType() == UniverseFactory::kUNIVERSEFACTORY_DEMO) {
        factory = std::make_unique<DemoUniverseFactory>();
    } else if (CommonServices::getConfig()->getUniverseFactoryType() == UniverseFactory::kUNIVERSEFACTORY_RANDOM) {
        factory = std::make_unique<RandomUniverseFactory>();
    } else {
        CommonServices::getExit()->ExitOnError();
    }

    GameWindow game_window;
    GameLoop game_loop(&game_window);
    game_window.Initialization();
    factory->GenerateBuffers();
    std::unique_ptr<Universe> universe{factory->GenerateUniverse()};

    /* Launch GL Scene */
    game_window.SetCamera((Camera*)universe->get_camera());
    game_loop.Loop(universe.get());

    CommonServices::getExit()->ExitOnSuccess();
}
