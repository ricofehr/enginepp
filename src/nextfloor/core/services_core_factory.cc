/**
 *  @file services_core_factory.c
 *  @brief Factory Class for common services
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/core/services_core_factory.h"

#include "nextfloor/core/file_config_parser.h"
#include "nextfloor/core/game_timer.h"
#include "nextfloor/core/program_exit.h"
#include "nextfloor/core/pseudo_random_generator.h"
#include "nextfloor/core/game_file_io.h"
#include "nextfloor/core/terminal_log.h"
#include "nextfloor/core/game_window_settings.h"

namespace nextfloor {

namespace core {

std::unique_ptr<ConfigParser> ServicesCoreFactory::MakeConfigParser() const
{
    return std::make_unique<FileConfigParser>();
}

std::unique_ptr<Exit> ServicesCoreFactory::MakeExit() const
{
    return std::make_unique<ProgramExit>();
}

std::unique_ptr<FileIO> ServicesCoreFactory::MakeFileIO() const
{
    return std::make_unique<GameFileIO>();
}

std::unique_ptr<nextfloor::gameplay::FrameTimer> ServicesCoreFactory::MakeFrameTimer() const
{
    return std::make_unique<GameTimer>();
}

std::unique_ptr<Log> ServicesCoreFactory::MakeLog() const
{
    return std::make_unique<TerminalLog>();
}

std::unique_ptr<RandomGenerator> ServicesCoreFactory::MakeRandomGenerator() const
{
    return std::make_unique<PseudoRandomGenerator>();
}

std::unique_ptr<WindowSettings> ServicesCoreFactory::MakeWindowSettings(nextfloor::gameplay::SceneWindow* sw) const
{
    return std::make_unique<GameWindowSettings>(sw);
}

}  // namespace core

}  // namespace nextfloor