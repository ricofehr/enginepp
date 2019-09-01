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
#include "nextfloor/core/timer.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"

namespace nextfloor {

namespace factory {

/**
 *  @class CoreFactory
 *  @brief Abstract Factory for common services
 */
class CoreFactory {

public:
    virtual ~CoreFactory() = default;

    virtual std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() = 0;
    virtual std::unique_ptr<nextfloor::core::Exit> MakeExit() = 0;
    virtual std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() = 0;
    virtual std::unique_ptr<nextfloor::core::Timer> MakeTimer() = 0;
    virtual std::unique_ptr<nextfloor::core::Log> MakeLog() = 0;
    virtual std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() = 0;

protected:
    CoreFactory() = default;

    CoreFactory(CoreFactory&&) = default;
    CoreFactory& operator=(CoreFactory&&) = default;
    CoreFactory(const CoreFactory&) = delete;
    CoreFactory& operator=(const CoreFactory&) = delete;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_COREFACTORY_H_
