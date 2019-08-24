/**
 *  @file serices_factory.h
 *  @brief Abstract Factory Class Implementation for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_SERVICESFACTORY_H_
#define NEXTFLOOR_FACTORY_SERVICESFACTORY_H_

#include "nextfloor/factory/core_factory.h"

#include <memory>

#include "nextfloor/core/config_parser.h"
#include "nextfloor/core/exit.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/timer.h"
#include "nextfloor/core/log.h"
#include "nextfloor/core/random_generator.h"
#include "nextfloor/factory/mesh_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class ServicesFactory
 *  @brief Abstract Factory for common services
 */
class ServicesFactory : CoreFactory {

public:
    ServicesFactory() = default;

    ServicesFactory(ServicesFactory&&) = default;
    ServicesFactory& operator=(ServicesFactory&&) = default;
    ServicesFactory(const ServicesFactory&) = default;
    ServicesFactory& operator=(const ServicesFactory&) = default;

    virtual ~ServicesFactory() override = default;

    virtual std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() const noexcept override;
    virtual std::unique_ptr<nextfloor::core::Exit> MakeExit() const noexcept override;
    virtual std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() const noexcept override;
    virtual std::unique_ptr<nextfloor::core::Timer> MakeTimer() const noexcept override;
    virtual std::unique_ptr<nextfloor::core::Log> MakeLog() const noexcept override;
    virtual std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() const noexcept override;
    virtual std::unique_ptr<MeshFactory> MakeMeshFactory() const noexcept override;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_SERVICESFACTORY_H_
