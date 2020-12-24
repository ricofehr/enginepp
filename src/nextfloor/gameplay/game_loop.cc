/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/gameplay/game_loop.h"

#include <cassert>
#include <sstream>

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace gameplay {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GameLoop::GameLoop(std::unique_ptr<Level> level,
                   SceneWindow* game_window,
                   std::unique_ptr<InputHandler> input_handler,
                   std::unique_ptr<FrameTimer> timer,
                   std::unique_ptr<Menu> main_menu)
{
    assert(!sInstanciated);
    sInstanciated = true;

    level_ = std::move(level);
    game_window_ = game_window;
    input_handler_ = std::move(input_handler);
    timer_ = std::move(timer);
    main_menu_ = std::move(main_menu);

    main_menu_->Init(game_window_->window());
}

void GameLoop::RunLoop()
{
    do {
        PollEvents();
        UpdateTime();
        LogLoop();
        ApplyLoop();
        CheckCurrentState();
    } while (IsInRunningState());
}


void GameLoop::ReturnToGame()
{
    main_menu_->Disable();
    input_handler_->ResetPointer();
    current_state_ = kInGameState;
}

void GameLoop::ExitGame()
{
    current_state_ = kExitState;
}

void GameLoop::ApplyLoop()
{
    if (IsInGame()) {
        RunGame();
    }
    else if (IsInMenu()) {
        DisplayMenu();
    }
}

void GameLoop::RunGame()
{
    UpdateCameraOrientation();
    HandlerInput();
    ManageCharacterStates();
    Draw();
}

void GameLoop::DisplayMenu()
{
    HandlerInput();
    game_window_->PrepareDisplay();
    level_->Draw(game_window_->getWindowRatio());
    main_menu_->MenuLoop();
    game_window_->SwapBuffers();
}

void GameLoop::UpdateTime()
{
    timer_->Loop();
}

void GameLoop::UpdateCameraOrientation()
{
    auto delta_angles = input_handler_->RecordHIDPointer(timer_->getDeltaTimeSinceLastLoop());
    level_->UpdateCameraOrientation(delta_angles);
}

void GameLoop::HandlerInput()
{
    auto command = input_handler_->HandlerInput();
    if (command) {
        level_->ExecutePlayerAction(command);
    }
}

void GameLoop::ManageCharacterStates()
{
    level_->UpdateCharacterStates(timer_->getDeltaTimeSinceLastLoop());
}

void GameLoop::Draw()
{
    game_window_->PrepareDisplay();
    level_->Move();
    level_->Draw(game_window_->getWindowRatio());
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
            message_frame << kMsInSecond / static_cast<double>(timer_->getLoopCountBySecond()) << " ms/frame - ";
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
    message_fps << timer_->getLoopCountBySecond() << " fps - ";
    CommonServices::getLog()->Write(std::move(message_fps));
}

void GameLoop::PollEvents()
{
    input_handler_->PollEvents();
}

void GameLoop::CheckCurrentState()
{
    if (current_state_ == kInGameState) {
        if (input_handler_->IsOpenMenuEventOccurs()) {
            SetMenuState();
        }
    }
    else
    {
        if (main_menu_->IsResumeGamePressed()) {
            ReturnToGame();
        }

        if (main_menu_->IsExitGamePressed()) {
            ExitGame();
        }
    }
}

void GameLoop::SetMenuState()
{
    main_menu_->Enable();
    current_state_ = kInMenuState;
}

GameLoop::~GameLoop() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace gameplay

}  // namespace nextfloor
