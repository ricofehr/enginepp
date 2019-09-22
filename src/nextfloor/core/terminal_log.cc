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

}  // anonymous namespace

TerminalLog::TerminalLog()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void TerminalLog::Write(std::ostringstream&& log_str) const
{
    std::cerr << log_str.str();
}

void TerminalLog::Write(const std::string& log_str) const
{
    std::cerr << log_str;
}

void TerminalLog::WriteLine(std::ostringstream&& log_line) const
{
    std::cerr << log_line.str() << std::endl;
}

void TerminalLog::WriteLine(const std::string& log_line) const
{
    std::cerr << log_line << std::endl;
}

TerminalLog::~TerminalLog() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace core

}  // namespace nextfloor
