/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>
#include <tbb/tbb.h>

#include "nextfloor/factory/factory.h"
#include "nextfloor/controller/game_loop.h"
#include "nextfloor/renderer/game_window.h"

#include "nextfloor/core/common_services.h"
#include "nextfloor/core/terminal_log.h"
#include "nextfloor/core/frame_timer.h"
#include "nextfloor/core/config_file.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/generic_file_io.h"
#include "nextfloor/core/standard_random_generator.h"

#include "nextfloor/objects/engine_camera.h"

int main(int argc, char* argv[])
{
    using nextfloor::objects::EngineCamera;
    using nextfloor::factory::Factory;
    using nextfloor::controller::GameLoop;
    using nextfloor::renderer::GameWindow;
    using nextfloor::core::CommonServices;
    using nextfloor::core::TerminalLog;
    using nextfloor::core::FrameTimer;
    using nextfloor::core::ConfigFile;
    using nextfloor::core::ProgramExit;
    using nextfloor::core::GenericFileIO;
    using nextfloor::core::StandardRandomGenerator;
    using nextfloor::objects::EngineCollision;
    using nextfloor::objects::EngineObject;

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
    std::unique_ptr<Factory> factory = std::make_unique<Factory>();
    CommonServices::provideFactory(factory.get());

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
    GameWindow game_window;
    std::unique_ptr<EngineCollision> engine_collision{factory->MakeCollisionEngine()};
    GameLoop game_loop(&game_window, engine_collision.get());
    game_window.Initialization();

    /* Launch GL Scene */
    std::unique_ptr<EngineObject> universe = factory->MakeDemoLevel()->GenerateUniverse();
    game_window.SetCamera(EngineCamera::active());
    game_loop.Loop(universe.get());

    CommonServices::getExit()->ExitOnSuccess();
}
