/**
 *  @file terminal_log.h
 *  @brief LOG output to current bash terminal
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_TERMINALLOG_H_
#define NEXTFLOOR_CORE_TERMINALLOG_H_

#include "nextfloor/core/log.h"

#include <iostream>
#include <sstream>


namespace nextfloor {

namespace core {

/**
 *  @class TerminalLog
 *  @brief Logging output to current bash terminal
 */
class TerminalLog : public Log {

public:
    TerminalLog();
    ~TerminalLog() noexcept final;

    void Write(std::ostringstream&& log_str) const final;
    void Write(const std::string& log_str) const final;

    void WriteLine(std::ostringstream&& log_line) const final;
    void WriteLine(const std::string& log_line) const final;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_TERMINALLOG_H_
