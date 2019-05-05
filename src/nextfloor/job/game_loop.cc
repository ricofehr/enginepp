/**
 *  @file game_loop.cc
 *  @brief GameLoop class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/job/game_loop.h"

#include <cassert>

#include "nextfloor/universe/universe.h"
#include "nextfloor/universe/dynamic/camera.h"
#include "nextfloor/hid/input_handler.h"
#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace job {

namespace {

static bool sInstanciated = false;

} // anonymous namespace

GameLoop::GameLoop(nextfloor::renderer::GameWindow* game_window)
{
    assert(!sInstanciated);
    sInstanciated = true;

    game_window_ = game_window;
    assert(game_window_ != nullptr);
}

/**
 *   Display global details for each seconds
 */
void GameLoop::LoopLog(nextfloor::universe::Universe* universe)
{
    using nextfloor::core::CommonServices;
    using nextfloor::graphics::Shape3D;

    static bool sFirstLoop = true;

    if (CommonServices::getTimer().IsNewSecondElapsed()) {
        int debug = CommonServices::getConfig().getSetting<int>("debug");

        /* Header for test datas output */
        if (sFirstLoop &&
            debug == CommonServices::getConfig().kDEBUG_TEST) {
            std::cout << "TIME:FPS:NBOBJALL:NBOBJMOVE" << std::endl;
        }
        /* Print if debug */
        if (debug == CommonServices::getConfig().kDEBUG_ALL) {
            std::cout << 1000.0 / static_cast<double>(CommonServices::getTimer().getFps()) << " ms/frame - ";
        }

        if (debug == CommonServices::getConfig().kDEBUG_PERF || debug == CommonServices::getConfig().kDEBUG_ALL) {
            std::cout << CommonServices::getTimer().getFps() << " fps (move facor: " << game_window_->getMoveFactor() << ") - ";
            std::cout << universe->countRoomsChilds(false) << " objects ("
                      << universe->countRoomsChilds(true) << " displayed) in ";
            std::cout << universe->countRooms(false) << " rooms (" << universe->countRooms(true) << " displayed)";
            std::cout << std::endl;
        }

        /* Test datas output */
        if (debug == CommonServices::getConfig().kDEBUG_TEST) {
            std::cout << CommonServices::getTimer().getFps() << ":";
            std::cout << universe->countRoomsChilds(true) << ":"
                      << universe->countRoomsMovingChilds(true) << std::endl;
        }

        /* First loop is ok */
        sFirstLoop = false;
    }
}

void GameLoop::Loop(nextfloor::universe::Universe* universe)
{
    using nextfloor::core::CommonServices;
    using nextfloor::universe::commands::Command;
    using nextfloor::universe::dynamic::Camera;
    using nextfloor::hid::InputHandler;

    InputHandler input_handler(game_window_->glfw_window());
    Camera* camera = game_window_->getCamera();

    /* Draw if window is focused and destroy window if ESC is pressed */
    do {
        CommonServices::getTimer().Loop();
        camera->ComputeFOV(input_handler.RecordFOV());
        camera->ComputeOrientation(input_handler.RecordHIDPointer());
        Command* command = input_handler.HandlerInput();
        if (command) {
            command->execute(camera);
        }

        if (CommonServices::getTimer().getFps() != 0) {
            game_window_->UpdateMoveFactor();
            universe->toready();
        }

        game_window_->PrepareDisplay();
        universe->Draw();
        game_window_->SwapBuffers();
        LoopLog(universe);

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
