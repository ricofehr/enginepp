/**
 *  @file global_log.h
 *  @brief LOG Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GLOBALLOG_H_
#define NEXTFLOOR_CORE_GLOBALLOG_H_

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::core
 *  @brief Core system
 */
namespace core {

/**
 *  @class GlobalLog
 *  @brief contains global centralized Operations for the engine
 */
class GlobalLog {

public:

    GlobalLog();

    GlobalLog(GlobalLog&&) = default;

    GlobalLog& operator=(GlobalLog&&) = default;

    /**
     *  Copy constructor Deleted : Ensure a sole Instance
     */
    GlobalLog(const GlobalLog&) = delete;

    /**
     *  Copy assignement Deleted: Ensure a sole Instance
     */
    GlobalLog& operator=(const GlobalLog&) = delete;

    ~GlobalLog();
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GLOBALLOG_H_