/**
 *  @file global_timer.h
 *  @brief Timer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GLOBALTIMER_H_
#define NEXTFLOOR_CORE_GLOBALTIMER_H_

#include <memory>
#include <chrono>

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
 *  @class GlobalTimer
 *  @brief GlobalTimer maintains global timer management for the program.
 */
class GlobalTimer {

public:

    GlobalTimer();

    /**
     *  Default Move constructor
     */
    GlobalTimer(GlobalTimer&&) = default;

    /**
     *  Default Move assignment
     */
    GlobalTimer& operator=(GlobalTimer&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    GlobalTimer(const GlobalTimer&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    GlobalTimer& operator=(const GlobalTimer&) = delete;

    ~GlobalTimer();

    /**
     *  Compute Times
     */
    void Loop() noexcept;

    /**
     *  Accessors
     */
    float getDeltaTimeSinceLastLoop() { return (float)deltatime_since_lastloop_; }
    int getFps() { return current_fps_; }
    bool IsNewSecondElapsed() { return is_newsecond_elapsed_; }

private:

    /**
     *  Compute FPS Count
     */
    void ComputeFps() noexcept;

    /**
     *  Check if program time is ending
     */
    void CheckEndProgram() noexcept;

    static double deltatime_since_lastloop_;

    static double totaltime_since_beginning_;

    static int current_fps_;

    /** True if current Loop is into a new elapsed second */
    static bool is_newsecond_elapsed_;

};


} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GLOBALTIMER_H_
