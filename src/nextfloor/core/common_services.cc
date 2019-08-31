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
    nextfloor::factory::ServicesFactory services_factory;

    config_ = services_factory.MakeConfigParser();
    file_io_ = services_factory.MakeFileIO();
    timer_ = services_factory.MakeTimer();
    log_ = services_factory.MakeLog();
    exit_ = services_factory.MakeExit();
    mesh_factory_ = services_factory.MakeMeshFactory();
    hid_factory_ = services_factory.MakeHidFactory();
    command_factory_ = services_factory.MakeCommandFactory();
}

}  // namespace core

}  // namespace nextfloor
