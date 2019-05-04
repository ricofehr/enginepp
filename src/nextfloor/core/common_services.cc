/**
 *  @file common_services.cc
 *  @brief CommonServices Implementation File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

GlobalConfig CommonServices::config_;
FileIO CommonServices::file_io_;
GlobalTimer CommonServices::timer_;
GlobalLog CommonServices::log_;
RandomNumber CommonServices::random_number_;
ExitProgram CommonServices::exit_;

} // namespace core

} // namespace nextfloor
