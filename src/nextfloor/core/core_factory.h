/**
 *  @file core_factory.h
 *  @brief Abstract Factory Class for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_COREFACTORY_H_
#define NEXTFLOOR_CORE_COREFACTORY_H_

#include <memory>

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/core/window_settings.h"

#include "nextfloor/gameplay/scene_window.h"

namespace nextfloor {

namespace core {

/**
 *  @class CoreFactory
 *  @brief Abstract Factory for common services
 */
class CoreFactory {

public:
    virtual ~CoreFactory() = default;

    virtual std::unique_ptr<ConfigParser> MakeConfigParser() const = 0;
    virtual std::unique_ptr<Exit> MakeExit() const = 0;
    virtual std::unique_ptr<FileIO> MakeFileIO() const = 0;
    virtual std::unique_ptr<Log> MakeLog() const = 0;
    virtual std::unique_ptr<RandomGenerator> MakeRandomGenerator() const = 0;
    virtual std::unique_ptr<WindowSettings> MakeWindowSettings(nextfloor::gameplay::SceneWindow* sw) const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_COREFACTORY_H_
