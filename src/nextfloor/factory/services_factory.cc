/**
 *  @file services_factory.c
 *  @brief Factory Class for common services
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/factory/services_factory.h"

#include "nextfloor/core/file_config_parser.h"
#include "nextfloor/core/game_timer.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/pseudo_random_generator.h"
#include "nextfloor/core/game_file_io.h"
#include "nextfloor/core/terminal_log.h"
#include "nextfloor/core/game_window_settings.h"

namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::core::ConfigParser> ServicesFactory::MakeConfigParser() const
{
    using nextfloor::core::FileConfigParser;
    return std::make_unique<FileConfigParser>();
}

std::unique_ptr<nextfloor::core::Exit> ServicesFactory::MakeExit() const
{
    using nextfloor::core::ProgramExit;
    return std::make_unique<ProgramExit>();
}

std::unique_ptr<nextfloor::core::FileIO> ServicesFactory::MakeFileIO() const
{
    using nextfloor::core::GameFileIO;
    return std::make_unique<GameFileIO>();
}

std::unique_ptr<nextfloor::gameplay::FrameTimer> ServicesFactory::MakeFrameTimer() const
{
    using nextfloor::core::GameTimer;
    return std::make_unique<GameTimer>();
}

std::unique_ptr<nextfloor::core::Log> ServicesFactory::MakeLog() const
{
    using nextfloor::core::TerminalLog;
    return std::make_unique<TerminalLog>();
}

std::unique_ptr<nextfloor::core::RandomGenerator> ServicesFactory::MakeRandomGenerator() const
{
    using nextfloor::core::PseudoRandomGenerator;
    return std::make_unique<PseudoRandomGenerator>();
}

std::unique_ptr<nextfloor::core::WindowSettings> ServicesFactory::MakeWindowSettings(nextfloor::gameplay::SceneWindow* sw) const
{
    using nextfloor::core::GameWindowSettings;
    return std::make_unique<GameWindowSettings>(sw);
}

}  // namespace factory

}  // namespace nextfloor