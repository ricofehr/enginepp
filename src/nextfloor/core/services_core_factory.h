/**
 *  @file serices_core_factory.h
 *  @brief Concrete Factory Class Implementation for core part
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_SERVICESCOREFACTORY_H_
#define NEXTFLOOR_CORE_SERVICESCOREFACTORY_H_

#include "nextfloor/core/core_factory.h"

namespace nextfloor {

namespace core {

/**
 *  @class ServicesCoreFactory
 *  @brief Concrete Factory for common services
 */
class ServicesCoreFactory : public CoreFactory {

public:
    std::unique_ptr<ConfigParser> MakeConfigParser() const final;
    std::unique_ptr<Exit> MakeExit() const final;
    std::unique_ptr<FileIO> MakeFileIO() const final;
    std::unique_ptr<Log> MakeLog() const final;
    std::unique_ptr<RandomGenerator> MakeRandomGenerator() const final;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_SERVICESCOREFACTORY_H_
