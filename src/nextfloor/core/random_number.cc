/**
 *  @file random_number.cc
 *  @brief Generate Random Numbers on the fly
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/random_number.h"

#include <cassert>
#include <time.h>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

RandomNumber::RandomNumber()
{
    assert(!sInstanciated);
    sInstanciated = true;

    /* Reset seed */
    srand(time(NULL));
}

RandomNumber::~RandomNumber()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor