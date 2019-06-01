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
 *  @brief Abstract class who defines Timer operations
 */
class Timer {

public:

    Timer(Timer&&) = default;

    Timer& operator=(Timer&&) = default;

    /* Copy constructor Deleted: Ensure a sole Instance */
    Timer(const Timer&) = delete;

    /* Copy assignment Deleted: Ensure a sole Instance */
    Timer& operator=(const Timer&) = delete;

    virtual ~Timer() = default;

    virtual void Loop() noexcept = 0;

    /**
     *  Accessors
     */
    virtual float getDeltaTimeSinceLastLoop() const = 0;
    virtual bool IsNewSecondElapsed() const = 0;
    virtual int getLoopCountBySecond() const = 0;

protected:

    Timer() = default;

private:

    virtual void ComputeTimers() noexcept = 0;
};


} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GLOBALTIMER_H_
