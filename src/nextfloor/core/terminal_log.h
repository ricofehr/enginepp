/**
 *  @file terminal_log.h
 *  @brief LOG output to current bash terminal
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_TERMINALLOG_H_
#define NEXTFLOOR_CORE_TERMINALLOG_H_

#include "nextfloor/core/log.h"

#include <iostream>


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

    /*  Copy constructor Deleted : Ensure a sole Instance */
    TerminalLog(const TerminalLog&) = delete;

    /* Copy assignement Deleted: Ensure a sole Instance */
    TerminalLog& operator=(const TerminalLog&) = delete;

    virtual ~TerminalLog() override final;

    virtual void Write(const std::string& log_line) override final;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_TERMINALLOG_H_
