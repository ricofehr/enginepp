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
    auto player = factory->MakePlayer(glm::vec3(0.0f, -2.0f, 5.0f));
    player_ = player.get();
    level_ = factory->MakeLevel();
    universe_ = level_->GenerateUniverse();
    universe_->AddIntoChild(std::move(player));
    game_cameras_ = universe_->all_cameras();
    SetActiveCamera(player_->camera());
    game_window_ = factory->MakeSceneWindow();
    input_handler_ = factory->MakeInputHandler();
}

void GameLoop::SetActiveCamera(nextfloor::objects::Camera* active_camera)
{
    std::list<nextfloor::objects::Camera*>::iterator it;
    for (it = game_cameras_.begin(); it != game_cameras_.end(); ++it) {
        if (*it == active_camera) {
            game_cameras_.remove(active_camera);
            game_cameras_.push_front(active_camera);
            break;
        }
    }
    universe_->set_active_camera(game_cameras_.front());
}

void GameLoop::Loop()
{
    do {
        UpdateTime();
        UpdateCameraOrientation();
        HandlerInput();
        DrawUniverse();
        LogLoop();

        input_handler_->PollEvents();
    } while (!input_handler_->IsCloseWindowEventOccurs());
}

void GameLoop::UpdateTime()
{
    timer_->Loop();
    if (timer_->getLoopCountBySecond() != 0) {
        game_window_->UpdateMoveFactor(timer_->getLoopCountBySecond());
        universe_->toready();
    }
}

void GameLoop::UpdateCameraOrientation()
{
    auto camera = game_cameras_.front();
    camera->ComputeFOV(input_handler_->RecordFOV());
    auto angles = input_handler_->RecordHIDPointer(timer_->getDeltaTimeSinceLastLoop());
    camera->increment_angles(angles.horizontal_delta_angle, angles.vertical_delta_angle);
    camera->ComputeOrientation();
}

void GameLoop::HandlerInput()
{
    auto command = input_handler_->HandlerInput();
    if (command) {
        command->execute(player_, timer_->getDeltaTimeSinceLastLoop());
    }
}

void GameLoop::DrawUniverse()
{
    game_window_->PrepareDisplay();
    universe_->Draw();
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

GameLoop::~GameLoop() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace gameplay

}  // namespace nextfloor
