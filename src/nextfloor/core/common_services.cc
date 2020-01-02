/**
 *  @file common_services.cc
 *  @brief CommonServices Implementation File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/common_services.h"

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

}  // namespace core

}  // namespace nextfloor
