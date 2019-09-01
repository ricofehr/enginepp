/**
 *  @file common_services.cc
 *  @brief CommonServices Implementation File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/common_services.h"

#include "nextfloor/factory/services_factory.h"

namespace nextfloor {

namespace core {

CommonServices::CommonServices()
{
    Init();
}

void CommonServices::Init()
{
    factory_ = std::make_unique<nextfloor::factory::FacadeFactory>();
    config_ = factory_->MakeConfigParser();
    file_io_ = factory_->MakeFileIO();
    timer_ = factory_->MakeTimer();
    log_ = factory_->MakeLog();
    exit_ = factory_->MakeExit();
}

}  // namespace core

}  // namespace nextfloor
