/**
 *  @file frame_timer.h
 *  @brief FrameTimer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_FRAMETIMER_H_
#define NEXTFLOOR_GAMEPLAY_FRAMETIMER_H_

namespace nextfloor {

namespace gameplay {

/**
 *  @class Timer
 *  @brief Pure interface who defines timer management between frames
 */
class FrameTimer {

public:
    virtual ~FrameTimer() = default;

    virtual void Loop() = 0;

    virtual float getDeltaTimeSinceLastLoop() const = 0;
    virtual bool IsNewSecondElapsed() const = 0;
    virtual int getLoopCountBySecond() const = 0;

private:
    virtual void ComputeTimers() = 0;
};


}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_FRAMETIMER_H_
