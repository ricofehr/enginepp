/**
 *  @file pseudo_random_generator.cc
 *  @brief Generate Random Numbers on the fly
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/pseudo_random_generator.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <ctime>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

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

PseudoRandomGenerator::~PseudoRandomGenerator() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace core

}  // namespace nextfloor
