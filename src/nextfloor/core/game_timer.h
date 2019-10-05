/**
 *  @file game_timer.h
 *  @brief GameTimer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GAMETIMER_H_
#define NEXTFLOOR_CORE_GAMETIMER_H_

#include "nextfloor/gameplay/frame_timer.h"

#include <memory>
#include <chrono>

namespace nextfloor {

namespace core {

/**
 *  @class GameTimer
 *  @brief GameTimer implements FrameTimer
 */
class GameTimer : public nextfloor::gameplay::FrameTimer {

public:
    GameTimer();
    ~GameTimer() noexcept final;

    void Loop() final;

    /**
     *  Accessors
     */
    inline float getDeltaTimeSinceLastLoop() const final { return static_cast<float>(deltatime_since_lastloop_); }

    int getLoopCountBySecond() const final { return current_fps_; }

    bool IsNewSecondElapsed() const final { return is_newsecond_elapsed_; }

private:
    void ComputeTimers() final;
    void ComputeFps();
    void CheckEndProgram();

    static double deltatime_since_lastloop_;
    static double totaltime_since_beginning_;
    static int current_fps_;

    /** True if current Loop is into a new elapsed second */
    static bool is_newsecond_elapsed_;
};


}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_GAMETIMER_H_
