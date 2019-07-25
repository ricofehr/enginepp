/**
 *  @file pseudo_random_generator.cc
 *  @brief Generate Random Numbers on the fly
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/pseudo_random_generator.h"

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <time.h>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

PseudoRandomGenerator::PseudoRandomGenerator()
{
    assert(!sInstanciated);
    sInstanciated = true;

    /* Reset seed */
    srand(static_cast<unsigned int>(time(nullptr)));
}

int PseudoRandomGenerator::GenerateNumber() const
{
    return rand();
}

PseudoRandomGenerator::~PseudoRandomGenerator()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor
