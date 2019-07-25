/**
 *  @file demo_level.cc
 *  @brief DemoLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/demo_level.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

std::unique_ptr<nextfloor::objects::EngineObject> DemoLevel::GenerateUniverse() noexcept
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeUniverse();
}

} // namespace gameplay

} // namespace nextfloor
