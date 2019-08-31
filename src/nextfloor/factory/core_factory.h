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

#include "nextfloor/factory/mesh_factory.h"
#include "nextfloor/factory/hid_factory.h"
#include "nextfloor/factory/command_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class CoreFactory
 *  @brief Abstract Factory for common services
 */
class CoreFactory {

public:
    virtual ~CoreFactory() = default;

    virtual std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::core::Exit> MakeExit() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::core::Timer> MakeTimer() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::core::Log> MakeLog() const noexcept = 0;
    virtual std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() const noexcept = 0;
    virtual std::unique_ptr<MeshFactory> MakeMeshFactory() const noexcept = 0;
    virtual std::unique_ptr<HidFactory> MakeHidFactory() const noexcept = 0;
    virtual std::unique_ptr<CommandFactory> MakeCommandFactory() const noexcept = 0;

protected:
    CoreFactory() = default;

    CoreFactory(CoreFactory&&) = default;
    CoreFactory& operator=(CoreFactory&&) = default;
    CoreFactory(const CoreFactory&) = default;
    CoreFactory& operator=(const CoreFactory&) = default;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_COREFACTORY_H_
