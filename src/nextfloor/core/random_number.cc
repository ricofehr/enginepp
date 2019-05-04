/**
 *  @file random_number.cc
 *  @brief Generate Random Numbers on the fly
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/random_number.h"

#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

RandomNumber::RandomNumber()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

RandomNumber::~RandomNumber()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor