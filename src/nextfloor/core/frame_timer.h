/**
 *  @file frame_timer.h
 *  @brief FrameTimer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FRAMETIMER_H_
#define NEXTFLOOR_CORE_FRAMETIMER_H_

#include "nextfloor/core/timer.h"

#include <memory>
#include <chrono>

namespace nextfloor {

namespace core {

/**
 *  @class FrameTimer
 *  @brief FrameTimer maintains timer management between frames
 */
class FrameTimer : public Timer {

public:
    FrameTimer();

    FrameTimer(FrameTimer&&) = default;
    FrameTimer& operator=(FrameTimer&&) = default;
    FrameTimer(const FrameTimer&) = delete;
    FrameTimer& operator=(const FrameTimer&) = delete;

    virtual ~FrameTimer() override final;

    virtual void Loop() noexcept override final;

    /**
     *  Accessors
     */
    inline virtual float getDeltaTimeSinceLastLoop() const override final
    {
        return static_cast<float>(deltatime_since_lastloop_);
    }

    virtual int getLoopCountBySecond() const override final { return current_fps_; }

    bool IsNewSecondElapsed() const override final { return is_newsecond_elapsed_; }

private:
    virtual void ComputeTimers() noexcept override final;
    void ComputeFps() noexcept;
    void CheckEndProgram() noexcept;

    static double deltatime_since_lastloop_;
    static double totaltime_since_beginning_;
    static int current_fps_;

    /** True if current Loop is into a new elapsed second */
    static bool is_newsecond_elapsed_;
};


}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_FRAMETIMER_H_
