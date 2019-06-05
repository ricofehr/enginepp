/**
 *  @file standard_random_generator.cc
 *  @brief Generate Random Numbers on the fly
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/standard_random_generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <time.h>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

StandardRandomGenerator::StandardRandomGenerator()
{
    assert(!sInstanciated);
    sInstanciated = true;

    /* Reset seed */
    srand(static_cast<unsigned int>(time(nullptr)));
}

int StandardRandomGenerator::GenerateNumber() const
{
    return rand();
}

StandardRandomGenerator::~StandardRandomGenerator()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor
