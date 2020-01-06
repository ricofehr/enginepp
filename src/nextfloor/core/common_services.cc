/**
 *  @file common_services.cc
 *  @brief CommonServices Implementation File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/common_services.h"

#include <cassert>

#include "nextfloor/core/services_core_factory.h"

namespace nextfloor {

namespace core {

CommonServices::CommonServices(const CoreFactory& factory)
{
    config_ = factory.MakeConfigParser();
    file_io_ = factory.MakeFileIO();
    log_ = factory.MakeLog();
    exit_ = factory.MakeExit();
    random_generator_ = factory.MakeRandomGenerator();
}

ConfigParser* CommonServices::getConfig()
{
    return Instance()->config();
}

const FileIO* CommonServices::getFileIO()
{
    return Instance()->fileIO();
}

const Log* CommonServices::getLog()
{
    return Instance()->log();
}

const RandomGenerator* CommonServices::getRandomGenerator()
{
    return Instance()->random_generator();
}

const Exit* CommonServices::getExit()
{
    return Instance()->exit();
}

CommonServices* CommonServices::Instance()
{
    static ServicesCoreFactory factory;
    /* Raw pointer because static var */
    static auto instance = new CommonServices(factory);
    return instance;
}

ConfigParser* CommonServices::config()
{
    assert(config_ != nullptr);
    return config_.get();
}

const FileIO* CommonServices::fileIO()
{
    assert(file_io_ != nullptr);
    return file_io_.get();
}

const Log* CommonServices::log() const
{
    assert(log_ != nullptr);
    return log_.get();
}

const RandomGenerator* CommonServices::random_generator() const
{
    assert(random_generator_ != nullptr);
    return random_generator_.get();
}

const Exit* CommonServices::exit() const
{
    assert(exit_ != nullptr);
    return exit_.get();
}

}  // namespace core

}  // namespace nextfloor
