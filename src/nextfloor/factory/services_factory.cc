/**
 *  @file services_factory.c
 *  @brief Factory Class for common services
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/services_factory.h"

#include "nextfloor/core/file_config_parser.h"
#include "nextfloor/core/frame_timer.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/pseudo_random_generator.h"
#include "nextfloor/core/std_file_io.h"
#include "nextfloor/core/terminal_log.h"

#include "nextfloor/factory/game_factory.h"
#include "nextfloor/factory/mouse_hid_factory.h"
#include "nextfloor/factory/action_factory.h"


namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::core::ConfigParser> ServicesFactory::MakeConfigParser() const noexcept
{
    using nextfloor::core::FileConfigParser;
    return std::make_unique<FileConfigParser>();
}

std::unique_ptr<nextfloor::core::Exit> ServicesFactory::MakeExit() const noexcept
{
    using nextfloor::core::ProgramExit;
    return std::make_unique<ProgramExit>();
}

std::unique_ptr<nextfloor::core::FileIO> ServicesFactory::MakeFileIO() const noexcept
{
    using nextfloor::core::StdFileIO;
    return std::make_unique<StdFileIO>();
}

std::unique_ptr<nextfloor::core::Timer> ServicesFactory::MakeTimer() const noexcept
{
    using nextfloor::core::FrameTimer;
    return std::make_unique<FrameTimer>();
}

std::unique_ptr<nextfloor::core::Log> ServicesFactory::MakeLog() const noexcept
{
    using nextfloor::core::TerminalLog;
    return std::make_unique<TerminalLog>();
}

std::unique_ptr<nextfloor::core::RandomGenerator> ServicesFactory::MakeRandomGenerator() const noexcept
{
    using nextfloor::core::PseudoRandomGenerator;
    return std::make_unique<PseudoRandomGenerator>();
}

std::unique_ptr<MeshFactory> ServicesFactory::MakeMeshFactory() const noexcept
{
    return std::make_unique<GameFactory>();
}

std::unique_ptr<HidFactory> ServicesFactory::MakeHidFactory() const noexcept
{
    return std::make_unique<MouseHidFactory>();
}

std::unique_ptr<CommandFactory> ServicesFactory::MakeCommandFactory() const noexcept
{
    return std::make_unique<ActionFactory>();
}

}  // namespace factory

}  // namespace nextfloor