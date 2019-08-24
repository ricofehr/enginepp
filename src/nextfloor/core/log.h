/**
 *  @file log.h
 *  @brief LOG Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_LOG_H_
#define NEXTFLOOR_CORE_LOG_H_

#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class Log
 *  @brief Abstract class who defines log operations
 */
class Log {

public:
    /*
     *  Debug Log Level
     */
    static constexpr int kDEBUG_QUIET = 0;
    static constexpr int kDEBUG_TEST = 1;
    static constexpr int kDEBUG_PERF = 2;
    static constexpr int kDEBUG_COLLISION = 3;
    static constexpr int kDEBUG_ALL = 4;

    Log(Log&&) = default;
    Log& operator=(Log&&) = default;
    Log(const Log&) = delete;
    Log& operator=(const Log&) = delete;

    virtual ~Log() = default;

    virtual void Write(std::ostringstream&& log_line) const = 0;
    virtual void Write(const std::string& log_line) const = 0;

    virtual void WriteLine(std::ostringstream&& log_line) const = 0;
    virtual void WriteLine(const std::string& log_line) const = 0;

protected:
    Log() = default;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_LOG_H_
