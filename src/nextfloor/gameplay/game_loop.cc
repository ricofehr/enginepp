/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/game_loop.h"

#include <cassert>
#include <sstream>
#include <list>

#include "nextfloor/gameplay/action.h"
#include "nextfloor/core/common_services.h"


#include "nextfloor/objects/model_mesh.h"

namespace nextfloor {

namespace gameplay {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GameLoop::GameLoop()
{
    assert(!sInstanciated);
    sInstanciated = true;

    auto factory = nextfloor::core::CommonServices::getFactory();
    timer_ = factory->MakeFrameTimer();
    level_ = factory->MakeLevel();
    game_window_ = factory->MakeSceneWindow();
    input_handler_ = factory->MakeInputHandler();
}

void GameLoop::Loop()
{
    do {
        UpdateTime();
        UpdateCameraOrientation();
        HandlerInput();
        Draw();
        LogLoop();
        PollEvents();
    } while (IsNextFrame());
}

void GameLoop::UpdateTime()
{
    timer_->Loop();
    if (timer_->getLoopCountBySecond() != 0) {
        game_window_->UpdateMoveFactor(timer_->getLoopCountBySecond());
        level_->toready();
    }
}

void GameLoop::UpdateCameraOrientation()
{
    auto delta_angles = input_handler_->RecordHIDPointer(timer_->getDeltaTimeSinceLastLoop());
    auto input_fov = input_handler_->RecordFOV();
    level_->UpdateCameraOrientation(delta_angles, input_fov);
}

void GameLoop::HandlerInput()
{
    auto command = input_handler_->HandlerInput();
    if (command) {
        level_->ExecutePlayerAction(command, timer_->getDeltaTimeSinceLastLoop());
    }
}

void GameLoop::Draw()
{
    game_window_->PrepareDisplay();
    level_->Move();
    level_->Draw();
    game_window_->SwapBuffers();
}

/**
 *   Display global details for each seconds
 */
void GameLoop::LogLoop()
{
    static bool sFirstLoop = true;

    using nextfloor::core::CommonServices;

    if (timer_->IsNewSecondElapsed()) {

        /* Header for test datas output */
        if (sFirstLoop && CommonServices::getConfig()->IsTestDebugEnabled()) {
            CommonServices::getLog()->Write("TIME:FPS:NBOBJALL:NBOBJMOVE");
        }
        /* Print if debug */
        if (CommonServices::getConfig()->IsAllDebugEnabled()) {
            std::ostringstream message_frame;
            message_frame << 1000.0 / static_cast<double>(timer_->getLoopCountBySecond()) << " ms/frame - ";
            CommonServices::getLog()->Write(std::move(message_frame));
        }

        if (CommonServices::getConfig()->IsPerfDebugEnabled()) {
            LogFps();
        }

        CommonServices::getLog()->WriteLine("");
        /* First loop is ok */
        sFirstLoop = false;
    }
}

void GameLoop::LogFps()
{
    using nextfloor::core::CommonServices;

    std::ostringstream message_fps;
    message_fps << timer_->getLoopCountBySecond();
    message_fps << " fps (move factor: " << game_window_->getFpsFixMoveFactor() << ") - ";
    CommonServices::getLog()->Write(std::move(message_fps));
}

void GameLoop::PollEvents()
{
    input_handler_->PollEvents();
}

bool GameLoop::IsNextFrame() const
{
    return !input_handler_->IsCloseWindowEventOccurs();
}

GameLoop::~GameLoop() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace gameplay

}  // namespace nextfloor
