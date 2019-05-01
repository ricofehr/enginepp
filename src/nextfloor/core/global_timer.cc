/**
 *  @file global_timer.cc
 *  @brief GlobalTimer Implementation File : Init global timers
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/global_timer.h"

namespace nextfloor {

namespace core {

double GlobalTimer::sDeltaTime = 0.0f;
double GlobalTimer::sTotalTime = 0.0f;

}

}