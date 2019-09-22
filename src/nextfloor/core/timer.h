/**
 *  @file timer.h
 *  @brief Timer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_TIMER_H_
#define NEXTFLOOR_CORE_TIMER_H_

namespace nextfloor {

namespace core {

/**
 *  @class Timer
 *  @brief Pure interface who defines Timer operations
 */
class Timer {

public:
    virtual ~Timer() = default;

    virtual void Loop() = 0;

    /**
     *  Accessors
     */
    virtual float getDeltaTimeSinceLastLoop() const = 0;
    virtual bool IsNewSecondElapsed() const = 0;
    virtual int getLoopCountBySecond() const = 0;

private:
    virtual void ComputeTimers() = 0;
};


}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_GLOBALTIMER_H_
