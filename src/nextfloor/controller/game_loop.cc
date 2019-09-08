/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/controller/game_loop.h"

#include <cassert>
#include <sstream>

#include "nextfloor/actions/action.h"
#include "nextfloor/core/common_services.h"


#include "nextfloor/objects/model_mesh.h"

namespace nextfloor {

namespace controller {

namespace {

static bool sInstanciated = false;

}  // anonymous namespace

GameLoop::GameLoop()
{
    assert(!sInstanciated);
    sInstanciated = true;

    using nextfloor::core::CommonServices;
    universe_ = CommonServices::getFactory()->MakeLevel()->GenerateUniverse();
    player_ = universe_->AddIntoChild(CommonServices::getFactory()->MakePlayer(glm::vec3(0.0f, -2.0f, 7.0f)));
    game_window_ = CommonServices::getFactory()->MakeSceneWindow();
    input_handler_ = CommonServices::getFactory()->MakeInputHandler();
    assert(game_window_ != nullptr);
}

/**
 *   Display global details for each seconds
 */
void GameLoop::LogLoop()
{
    static bool sFirstLoop = true;

    using nextfloor::core::CommonServices;

    if (CommonServices::getTimer()->IsNewSecondElapsed()) {

        auto rooms = universe_->childs();
        for (auto& r : rooms) {
            if (r->IsInside(player_)) {
                dynamic_cast<nextfloor::objects::ModelMesh*>(r)->grid()->DisplayGrid();
            }
        }

        /* Header for test datas output */
        if (sFirstLoop && CommonServices::getConfig()->IsTestDebugEnabled()) {
            CommonServices::getLog()->Write("TIME:FPS:NBOBJALL:NBOBJMOVE");
        }
        /* Print if debug */
        if (CommonServices::getConfig()->IsAllDebugEnabled()) {
            std::ostringstream message_frame;
            message_frame << 1000.0 / static_cast<double>(CommonServices::getTimer()->getLoopCountBySecond())
                          << " ms/frame - ";
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
    message_fps << CommonServices::getTimer()->getLoopCountBySecond();
    message_fps << " fps (move factor: " << game_window_->getFpsFixMoveFactor() << ") - ";
    CommonServices::getLog()->Write(std::move(message_fps));
}

void GameLoop::Loop()
{
    using nextfloor::core::CommonServices;

    /* Draw if window is focused and destroy window if ESC is pressed */
    do {
        CommonServices::getTimer()->Loop();

        if (CommonServices::getTimer()->getLoopCountBySecond() != 0) {
            game_window_->UpdateMoveFactor();
            universe_->toready();
        }

        auto camera = player_->camera();
        camera->ComputeFOV(input_handler_->RecordFOV());
        auto angles = input_handler_->RecordHIDPointer();
        camera->increment_angles(angles.horizontal_delta_angle, angles.vertical_delta_angle);
        camera->ComputeOrientation();
        auto command = input_handler_->HandlerInput();
        if (command) {
            command->execute(player_);
        }

        game_window_->PrepareDisplay();
        auto rooms = universe_->childs();
        for (auto& r : rooms) {
            if (r->IsInside(player_)) {
                r->DetectCollision();
                r->Move();
                r->Draw();
            }
        }
        // universe_->DetectCollision();
        // universe_->Move();
        // universe_->Draw();
        game_window_->SwapBuffers();
        LogLoop();

        input_handler_->PollEvents();
    } while (!input_handler_->IsCloseWindowEventOccurs());
}

GameLoop::~GameLoop()
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace controller

}  // namespace nextfloor
