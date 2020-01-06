/**
 *  @file log.h
 *  @brief LOG Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_LOG_H_
#define NEXTFLOOR_CORE_LOG_H_

#include <string>
#include <sstream>

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
    static constexpr int kDebugQuiet = 0;
    static constexpr int kDebugTest = 1;
    static constexpr int kDebugPerf = 2;
    static constexpr int kDebugCollision = 3;
    static constexpr int kDebugAll = 4;

    virtual ~Log() = default;

    virtual void Write(std::ostringstream&& log_line) const = 0;
    virtual void Write(const std::string& log_line) const = 0;

    virtual void WriteLine(std::ostringstream&& log_line) const = 0;
    virtual void WriteLine(const std::string& log_line) const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_LOG_H_
