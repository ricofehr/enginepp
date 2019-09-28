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
    std::unique_ptr<nextfloor::core::ConfigParser> MakeConfigParser() const final;
    std::unique_ptr<nextfloor::core::Exit> MakeExit() const final;
    std::unique_ptr<nextfloor::core::FileIO> MakeFileIO() const final;
    std::unique_ptr<nextfloor::core::FrameTimer> MakeFrameTimer() const final;
    std::unique_ptr<nextfloor::core::Log> MakeLog() const final;
    std::unique_ptr<nextfloor::core::RandomGenerator> MakeRandomGenerator() const final;
    std::unique_ptr<nextfloor::core::WindowSettings> MakeWindowSettings(nextfloor::renderer::SceneWindow* sw) const final;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_SERVICESFACTORY_H_
