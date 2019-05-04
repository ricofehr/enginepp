/**
 *  @file exit_program.cc
 *  @brief Manage exits program
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/exit_program.h"

#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

ExitProgram::ExitProgram()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

ExitProgram::~ExitProgram()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor