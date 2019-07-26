/**
 *  @file nextfloor.cc
 *  @brief Main Function File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include <memory>
#include <tbb/tbb.h>

#include "nextfloor/factory/game_factory.h"
#include "nextfloor/controller/game_loop.h"
#include "nextfloor/renderer/game_window.h"

#include "nextfloor/core/common_services.h"
#include "nextfloor/core/terminal_log.h"
#include "nextfloor/core/frame_timer.h"
#include "nextfloor/core/file_config.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/std_file_io.h"
#include "nextfloor/core/pseudo_random_generator.h"

#include "nextfloor/objects/camera.h"

int main(int argc, char* argv[])
{
    using nextfloor::objects::Camera;
    using nextfloor::factory::GameFactory;
    using nextfloor::controller::GameLoop;
    using nextfloor::renderer::GameWindow;
    using nextfloor::core::CommonServices;
    using nextfloor::core::TerminalLog;
    using nextfloor::core::FrameTimer;
    using nextfloor::core::FileConfig;
    using nextfloor::core::ProgramExit;
    using nextfloor::core::StdFileIO;
    using nextfloor::core::PseudoRandomGenerator;
    using nextfloor::objects::CollisionEngine;
    using nextfloor::objects::Mesh;

    /* Init CommonServices */
    std::unique_ptr<TerminalLog> terminal_log = std::make_unique<TerminalLog>();
    CommonServices::provideLog(terminal_log.get());
    std::unique_ptr<FrameTimer> frame_timer  = std::make_unique<FrameTimer>();
    CommonServices::provideTimer(frame_timer.get());
    std::unique_ptr<FileConfig> file_config = std::make_unique<FileConfig>();
    CommonServices::provideConfig(file_config.get());
    std::unique_ptr<ProgramExit> program_exit = std::make_unique<ProgramExit>();
    CommonServices::provideExit(program_exit.get());
    std::unique_ptr<StdFileIO> generic_file_io = std::make_unique<StdFileIO>();
    CommonServices::provideFileIO(generic_file_io.get());
    std::unique_ptr<PseudoRandomGenerator> standard_random_generator = std::make_unique<PseudoRandomGenerator>();
    CommonServices::provideRandomGenerator(standard_random_generator.get());
    std::unique_ptr<GameFactory> factory = std::make_unique<GameFactory>();
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
    std::unique_ptr<CollisionEngine> engine_collision{factory->MakeCollisionEngine()};
    GameLoop game_loop(&game_window, engine_collision.get());
    game_window.Initialization();

    /* Launch GL Scene */
    std::unique_ptr<Mesh> universe = factory->MakeLevel()->GenerateUniverse();
    game_window.SetCamera(Camera::active());
    game_loop.Loop(universe.get());

    CommonServices::getExit()->ExitOnSuccess();
}
