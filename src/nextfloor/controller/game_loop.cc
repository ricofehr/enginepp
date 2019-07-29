/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/controller/game_loop.h"

#include <cassert>

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
void GameLoop::LoopLog()
{
    using nextfloor::core::CommonServices;

    static bool sFirstLoop = true;

    if (CommonServices::getTimer()->IsNewSecondElapsed()) {
        int debug = CommonServices::getConfig()->getDebugLevel();

        /* Header for test datas output */
        if (sFirstLoop &&
            debug == CommonServices::getLog()->kDEBUG_TEST) {
            std::cout << "TIME:FPS:NBOBJALL:NBOBJMOVE" << std::endl;
        }
        /* Print if debug */
        if (debug == CommonServices::getLog()->kDEBUG_ALL) {
            std::cout << 1000.0 / static_cast<double>(CommonServices::getTimer()->getLoopCountBySecond()) << " ms/frame - ";
        }

        if (debug == CommonServices::getLog()->kDEBUG_PERF || debug == CommonServices::getLog()->kDEBUG_ALL) {
            std::cout << CommonServices::getTimer()->getLoopCountBySecond() << " fps (move facor: " << game_window_->getMoveFactor() << ") - ";
            std::cout << std::endl;
        }

        /* Test datas output */
        if (debug == CommonServices::getLog()->kDEBUG_TEST) {
            std::cout << CommonServices::getTimer()->getLoopCountBySecond() << ":";
        }

        /* First loop is ok */
        sFirstLoop = false;
    }
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
        LoopLog();

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
