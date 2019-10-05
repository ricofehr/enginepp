/**
 *  @file core_factory.h
 *  @brief Abstract Factory Class for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_COREFACTORY_H_
#define NEXTFLOOR_FACTORY_COREFACTORY_H_

#include <memory>

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/gameplay/frame_timer.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/core/window_settings.h"

#include "nextfloor/gameplay/scene_window.h"

namespace nextfloor {

namespace factory {

/**
 *  @class CoreFactory
 *  @brief Abstract Factory for common services
 */
class CoreFactory {

public:
    virtual ~CoreFactory() = default;

    virtual std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() const = 0;
    virtual std::unique_ptr<nextfloor::core::Exit> MakeExit() const = 0;
    virtual std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() const = 0;
    virtual std::unique_ptr<nextfloor::gameplay::FrameTimer> MakeFrameTimer() const = 0;
    virtual std::unique_ptr<nextfloor::core::Log> MakeLog() const = 0;
    virtual std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() const = 0;
    virtual std::unique_ptr<nextfloor::core::WindowSettings>
      MakeWindowSettings(nextfloor::gameplay::SceneWindow* sw) const = 0;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_COREFACTORY_H_
