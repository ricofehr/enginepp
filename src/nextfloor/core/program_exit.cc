/**
 *  @file exit_program.cc
 *  @brief Manage exits program
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/program_exit.h"

#include <stdlib.h>
#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

ProgramExit::ProgramExit()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void ProgramExit::ExitOnError() const
{
    exit(1);
}

void ProgramExit::ExitOnSuccess() const
{
    exit(0);
}

ProgramExit::~ProgramExit() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace core

}  // namespace nextfloor
