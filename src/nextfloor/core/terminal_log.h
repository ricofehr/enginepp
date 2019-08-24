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

    TerminalLog(TerminalLog&&) = default;
    TerminalLog& operator=(TerminalLog&&) = default;
    TerminalLog(const TerminalLog&) = delete;
    TerminalLog& operator=(const TerminalLog&) = delete;

    virtual ~TerminalLog() override final;

    virtual void Write(std::ostringstream&& log_str) const override final;
    virtual void Write(const std::string& log_str) const override final;

    virtual void WriteLine(std::ostringstream&& log_line) const override final;
    virtual void WriteLine(const std::string& log_line) const override final;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_TERMINALLOG_H_
