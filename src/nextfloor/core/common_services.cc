/**
 *  @file common_services.cc
 *  @brief CommonServices Implementation File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

Config* CommonServices::config_ = nullptr;
FileIO* CommonServices::file_io_ = nullptr;
Timer* CommonServices::timer_ = nullptr;
Log* CommonServices::log_ = nullptr;
RandomGenerator* CommonServices::random_generator_ = nullptr;
Exit* CommonServices::exit_ = nullptr;

} // namespace core

} // namespace nextfloor
