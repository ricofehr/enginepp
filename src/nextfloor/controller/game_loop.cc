/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/controller/game_loop.h"

#include <cassert>
#include <sstream>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace controller {

namespace {

static bool sInstanciated = false;

} // anonymous namespace

GameLoop::GameLoop()
{
    assert(!sInstanciated);
    sInstanciated = true;

    using nextfloor::core::CommonServices;
    game_window_ = CommonServices::getFactory()->MakeSceneWindow();
    engine_collision_ = CommonServices::getFactory()->MakeCollisionEngine();
    universe_ = CommonServices::getFactory()->MakeLevel()->GenerateUniverse();

    assert(game_window_ != nullptr);
    assert(engine_collision_ != nullptr);
}

/**
 *   Display global details for each seconds
 */
void GameLoop::LogLoop()
{
    using nextfloor::core::CommonServices;

    static bool sFirstLoop = true;

    if (CommonServices::getTimer()->IsNewSecondElapsed()) {
        /* Header for test datas output */
        if (sFirstLoop && CommonServices::getConfig()->IsTestDebugEnabled()) {
            CommonServices::getLog()->Write("TIME:FPS:NBOBJALL:NBOBJMOVE");
        }
        /* Print if debug */
        if (CommonServices::getConfig()->IsAllDebugEnabled()) {
            std::ostringstream message_frame;
            message_frame << 1000.0 / static_cast<double>(CommonServices::getTimer()->getLoopCountBySecond()) << " ms/frame - ";
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
    message_fps << " fps (move factor: " << game_window_->getMoveFactor() << ") - ";
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

        game_window_->PrepareDisplay();
        universe_->Draw();
        game_window_->SwapBuffers();
        LogLoop();

        glfwPollEvents();
    }
    while (glfwGetKey(game_window_->glfw_window(), GLFW_KEY_ESCAPE) != GLFW_PRESS
           && glfwWindowShouldClose(game_window_->glfw_window()) == 0);
}

GameLoop::~GameLoop()
{
    assert(sInstanciated);
    sInstanciated = false;
}

} // namespace renderer

} // namespace nextfloor
