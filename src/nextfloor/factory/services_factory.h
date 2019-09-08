/**
 *  @file serices_factory.h
 *  @brief Abstract Factory Class Implementation for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_SERVICESFACTORY_H_
#define NEXTFLOOR_FACTORY_SERVICESFACTORY_H_

#include "nextfloor/factory/core_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class ServicesFactory
 *  @brief Abstract Factory for common services
 */
class ServicesFactory : public CoreFactory {

public:
    ServicesFactory() = default;

    ServicesFactory(ServicesFactory&&) = default;
    ServicesFactory& operator=(ServicesFactory&&) = default;
    ServicesFactory(const ServicesFactory&) = delete;
    ServicesFactory& operator=(const ServicesFactory&) = delete;

    virtual ~ServicesFactory() override = default;

    virtual std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() override;
    virtual std::unique_ptr<nextfloor::core::Exit> MakeExit() override;
    virtual std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() override;
    virtual std::unique_ptr<nextfloor::core::Timer> MakeTimer() override;
    virtual std::unique_ptr<nextfloor::core::Log> MakeLog() override;
    virtual std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() override;
    virtual std::unique_ptr<nextfloor::core::WindowSettings> MakeWindowSettings(nextfloor::renderer::SceneWindow* sw) override;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_SERVICESFACTORY_H_
