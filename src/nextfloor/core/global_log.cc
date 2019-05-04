/**
 *  @file global_log.cc
 *  @brief Log Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/global_log.h"

#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

GlobalLog::GlobalLog()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

GlobalLog::~GlobalLog()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor