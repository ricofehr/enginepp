/**
 *  @file input_handler.cc
 *  @brief InputHandler Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/game_input_handler.h"

#include <memory>

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/renderer/scene_input.h"

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace hid {

/**
 *  Constructor
 *  @param  window  GL Main Window
 */
GameInputHandler::GameInputHandler()
{
    /** TODO: init hid in external function with config check for targetted controller */
    hid_ = std::make_unique<MouseKeyboard>();

    /* Init Command Pointers */
    auto factory = nextfloor::core::CommonServices::getFactory();
    move_up_command_ = factory->MakeMoveUpAction();
    move_down_command_ = factory->MakeMoveDownAction();
    move_left_command_ = factory->MakeMoveLeftAction();
    move_right_command_ = factory->MakeMoveRightAction();
    jump_command_ = factory->MakeJumpAction();
    run_command_ = factory->MakeRunAction();
    fire_command_ = factory->MakeFireAction();
}

nextfloor::gameplay::Action* GameInputHandler::HandlerInput()
{
    using nextfloor::renderer::SceneInput;

    if (hid_->isPressed(SceneInput::kINPUT_UP)) {
        return move_up_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_DOWN)) {
        return move_down_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_LEFT)) {
        return move_left_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_RIGHT)) {
        return move_right_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_JUMP)) {
        return jump_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_FIRE)) {
        return fire_command_.get();
    }

    if (hid_->isPressed(SceneInput::kINPUT_RUN)) {
        return run_command_.get();
    }

    return nullptr;
}

nextfloor::gameplay::HIDPointer GameInputHandler::RecordHIDPointer(double elapsed_time)
{
    return hid_->RecordHIDPointer(elapsed_time);
}

float GameInputHandler::RecordFOV()
{
    return hid_->RecordFOV();
}

void GameInputHandler::PollEvents()
{
    hid_->PollEvents();
}

bool GameInputHandler::IsCloseWindowEventOccurs()
{
    return hid_->IsCloseWindowEventOccurs();
}

}  // namespace hid

}  // namespace nextfloor