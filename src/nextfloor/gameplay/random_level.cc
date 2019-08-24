/**
 *  @file random_level.cc
 *  @brief RandomLevel class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/random_level.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

std::unique_ptr<nextfloor::objects::Mesh> RandomLevel::GenerateUniverse() noexcept
{
    using nextfloor::core::CommonServices;
    return CommonServices::getFactory()->MakeUniverse();
}

}  // namespace gameplay

}  // namespace nextfloor
