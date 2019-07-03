/**
 *  @file terminal_log.cc
 *  @brief LOG output to current bash terminal
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/terminal_log.h"

#include <cassert>

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

TerminalLog::TerminalLog()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void TerminalLog::Write(const std::string& log_line)
{
    std::cerr << log_line << std::endl;
}

TerminalLog::~TerminalLog()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor
