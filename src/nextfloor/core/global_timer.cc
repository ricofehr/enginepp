/**
 *  @file global_timer.cc
 *  @brief GlobalTimer Implementation File : Init global timers
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/global_timer.h"

#include "nextfloor/core/config_engine.h"

namespace nextfloor {

namespace core {

double GlobalTimer::sDeltaTime = 0.0f;
double GlobalTimer::sTotalTime = 0.0f;
int GlobalTimer::sFps = 0;
bool GlobalTimer::sSecondElapsed = false;

/**
 *  Compute Time
 */
void GlobalTimer::Loop() noexcept
{
    using namespace std::chrono;
    static double sLastTime = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();

    /* Delta time */
    double current_time = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
    sDeltaTime = current_time - sLastTime;
    sLastTime = current_time;
    sTotalTime += sDeltaTime;

    ComputeFps();
    CheckEndProgram();
}

/**
 *  Compute FPS Count
 */
void GlobalTimer::ComputeFps() noexcept
{
    static int sLastSecondTime = (int)GlobalTimer::sTotalTime;
    static int sNbFrames = 0;

    sSecondElapsed = false;
    sNbFrames++;

    if (GlobalTimer::sTotalTime - (double)sLastSecondTime >= 1.0) {
        sFps = sNbFrames;
        /* Reset timer */
        sNbFrames = 0;
        sLastSecondTime += 1;
        sSecondElapsed = true;
    }
}

/**
 *  Check if program time is ending
 */
void GlobalTimer::CheckEndProgram() noexcept
{
    int end_time = ConfigEngine::getSetting<int>("execution_time");
    if (end_time && GlobalTimer::sTotalTime >= end_time) {
        exit(0);
    }
}


} // namespace core

} // namespace nextfloor