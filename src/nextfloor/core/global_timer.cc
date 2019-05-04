/**
 *  @file global_timer.cc
 *  @brief GlobalTimer Implementation File : Init global timers
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/global_timer.h"

#include <cassert>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

}

double GlobalTimer::deltatime_since_lastloop_ = 0.0f;
double GlobalTimer::totaltime_since_beginning_ = 0.0f;
int GlobalTimer::current_fps_ = 0;
bool GlobalTimer::is_newsecond_elapsed_ = false;

GlobalTimer::GlobalTimer()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void GlobalTimer::Loop() noexcept
{
    using namespace std::chrono;
    static double sLastTime = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();

    /* Delta time */
    double current_time = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    deltatime_since_lastloop_ = current_time - sLastTime;
    sLastTime = current_time;
    totaltime_since_beginning_ += deltatime_since_lastloop_;

    ComputeFps();
    CheckEndProgram();
}

void GlobalTimer::ComputeFps() noexcept
{
    static int sLastSecondTime = (int)totaltime_since_beginning_;
    static int sNbFrames = 0;

    is_newsecond_elapsed_ = false;
    sNbFrames++;

    if (totaltime_since_beginning_ - (double)sLastSecondTime >= 1.0) {
        current_fps_ = sNbFrames;
        /* Reset timer */
        sNbFrames = 0;
        sLastSecondTime += 1;
        is_newsecond_elapsed_ = true;
    }
}

void GlobalTimer::CheckEndProgram() noexcept
{
    int end_time = CommonServices::getConfig().getSetting<int>("execution_time");
    if (end_time && totaltime_since_beginning_ >= end_time) {
        exit(0);
    }
}

GlobalTimer::~GlobalTimer()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace core

} // namespace nextfloor