/**
 *  @file frame_timer.cc
 *  @brief FrameTimer Implementation File : Init frame timers
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/frame_timer.h"

#include <cassert>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

static double GetNow()
{
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

}

double FrameTimer::deltatime_since_lastloop_ = 0.0f;
double FrameTimer::totaltime_since_beginning_ = 0.0f;
int FrameTimer::current_fps_ = 0;
bool FrameTimer::is_newsecond_elapsed_ = false;

FrameTimer::FrameTimer()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void FrameTimer::Loop() noexcept
{
    ComputeTimers();
    ComputeFps();
    CheckEndProgram();
}

void FrameTimer::ComputeTimers() noexcept
{
    static double sLastTime = GetNow();

    double current_time = GetNow();
    deltatime_since_lastloop_ = current_time - sLastTime;
    sLastTime = current_time;
    totaltime_since_beginning_ += deltatime_since_lastloop_;
}

void FrameTimer::ComputeFps() noexcept
{
    static int sLastSecondTime = static_cast<int>(totaltime_since_beginning_);
    static int sNbFrames = 0;

    is_newsecond_elapsed_ = false;
    sNbFrames++;

    if (totaltime_since_beginning_ - static_cast<double>(sLastSecondTime) >= 1.0) {
        current_fps_ = sNbFrames;
        /* Reset timer */
        sNbFrames = 0;
        sLastSecondTime += 1;
        is_newsecond_elapsed_ = true;
    }
}

void FrameTimer::CheckEndProgram() noexcept
{
    int end_time = CommonServices::getConfig()->getExecutionDuration();
    if (end_time && totaltime_since_beginning_ >= end_time) {
        CommonServices::getExit()->ExitOnSuccess();
    }
}

FrameTimer::~FrameTimer()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor
