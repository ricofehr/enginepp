/**
 *  @file game_timer.cc
 *  @brief GameTimer Implementation File : Init frame timers
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/game_timer.h"

#include <cassert>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace gameplay {

namespace {

static bool sInstanciated = false;

static double GetNow()
{
    using namespace std::chrono;
    return duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
}

}  // anonymous namespace

double GameTimer::deltatime_since_lastloop_ = 0.0f;
double GameTimer::totaltime_since_beginning_ = 0.0f;
int GameTimer::current_fps_ = 0;
bool GameTimer::is_newsecond_elapsed_ = false;

GameTimer::GameTimer()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void GameTimer::Loop()
{
    ComputeTimers();
    ComputeFps();
    CheckEndProgram();
}

void GameTimer::ComputeTimers()
{
    static double sLastTime = GetNow();

    double current_time = GetNow();
    deltatime_since_lastloop_ = current_time - sLastTime;
    sLastTime = current_time;
    totaltime_since_beginning_ += deltatime_since_lastloop_;
}

void GameTimer::ComputeFps()
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

void GameTimer::CheckEndProgram()
{
    using nextfloor::core::CommonServices;

    int end_time = CommonServices::getConfig()->getExecutionDuration();
    if (end_time && totaltime_since_beginning_ >= end_time) {
        CommonServices::getExit()->ExitOnSuccess();
    }
}

GameTimer::~GameTimer() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace gameplay

}  // namespace nextfloor
