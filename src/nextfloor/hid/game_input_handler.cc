/**
 *  @file input_handler.cc
 *  @brief InputHandler Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/game_input_handler.h"

#include <memory>

#include "nextfloor/hid/mouse_keyboard.h"

#include "nextfloor/actions/moveup_action.h"
#include "nextfloor/actions/movedown_action.h"
#include "nextfloor/actions/moveleft_action.h"
#include "nextfloor/actions/moveright_action.h"
#include "nextfloor/actions/jump_action.h"
#include "nextfloor/actions/run_action.h"
#include "nextfloor/actions/fire_action.h"

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
    move_up_command_ = std::make_unique<nextfloor::actions::MoveUpAction>();
    move_down_command_ = std::make_unique<nextfloor::actions::MoveDownAction>();
    move_left_command_ = std::make_unique<nextfloor::actions::MoveLeftAction>();
    move_right_command_ = std::make_unique<nextfloor::actions::MoveRightAction>();
    jump_command_ = std::make_unique<nextfloor::actions::JumpAction>();
    run_command_ = std::make_unique<nextfloor::actions::RunAction>();
    fire_command_ = std::make_unique<nextfloor::actions::FireAction>();
}

nextfloor::actions::Action* GameInputHandler::HandlerInput()
{
    if (hid_->isPressed(HID::kHID_UP)) {
        return move_up_command_.get();
    }

    if (hid_->isPressed(HID::kHID_DOWN)) {
        return move_down_command_.get();
    }

    if (hid_->isPressed(HID::kHID_LEFT)) {
        return move_left_command_.get();
    }

    if (hid_->isPressed(HID::kHID_RIGHT)) {
        return move_right_command_.get();
    }

    if (hid_->isPressed(HID::kHID_JUMP)) {
        return jump_command_.get();
    }

    if (hid_->isPressed(HID::kHID_FIRE)) {
        return fire_command_.get();
    }

    if (hid_->isPressed(HID::kHID_RUN)) {
        return run_command_.get();
    }

    return nullptr;
}

HIDPointer GameInputHandler::RecordHIDPointer()
{
    return hid_->RecordHIDPointer();
}

float GameInputHandler::RecordFOV()
{
    return hid_->RecordFOV();
}

}  // namespace hid

}  // namespace nextfloor