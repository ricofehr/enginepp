/**
 *  @file input_handler.cc
 *  @brief InputHandler Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/input_handler.h"

#include <memory>

#include "nextfloor/hid/keyboard_mouse.h"

#include "nextfloor/universe/commands/moveup_command.h"
#include "nextfloor/universe/commands/movedown_command.h"
#include "nextfloor/universe/commands/moveleft_command.h"
#include "nextfloor/universe/commands/moveright_command.h"
#include "nextfloor/universe/commands/jump_command.h"
#include "nextfloor/universe/commands/run_command.h"
#include "nextfloor/universe/commands/fire_command.h"

namespace nextfloor {

namespace hid {

/**
 *  Constructor
 *  @param  window  GL Main Window
 */
InputHandler::InputHandler(GLFWwindow* window)
{
    window_ = window;

    /** TODO: init hid in external function with config check for targetted controller */
    hid_ = std::make_unique<KeyboardMouse>();

    /* Ensure we can capture keys being pressed below */
    glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);

    /* Init Command Pointers */
    move_up_command_ = std::make_unique<nextfloor::universe::commands::MoveUpCommand>();
    move_down_command_ = std::make_unique<nextfloor::universe::commands::MoveDownCommand>();
    move_left_command_ = std::make_unique<nextfloor::universe::commands::MoveLeftCommand>();
    move_right_command_ = std::make_unique<nextfloor::universe::commands::MoveRightCommand>();
    jump_command_ = std::make_unique<nextfloor::universe::commands::JumpCommand>();
    run_command_ = std::make_unique<nextfloor::universe::commands::RunCommand>();
    fire_command_ = std::make_unique<nextfloor::universe::commands::FireCommand>();
}

/**
 *  Get Current State Input
 *  @return Command Object
 */
nextfloor::universe::commands::Command* InputHandler::HandlerInput() noexcept
{
    if (hid_->isPressed(window_, HID::kHID_UP)) {
        return move_up_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_DOWN)) {
        return move_down_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_LEFT)) {
        return move_left_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_RIGHT)) {
        return move_right_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_JUMP)) {
        return jump_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_FIRE)) {
        return fire_command_.get();
    }

    if (hid_->isPressed(window_, HID::kHID_RUN)) {
        return run_command_.get();
    }

    return nullptr;
}

/**
 *  Get HID Pointer angles changes
 *  @return HIDPointer struct
 */
HIDPointer InputHandler::RecordHIDPointer() noexcept
{
    return hid_->RecordHIDPointer(window_);
}

/**
 *  Get FOV (FieldOfView) Current State
 *  @return FOV value
 */
float InputHandler::RecordFOV() noexcept
{
    return hid_->RecordFOV(window_);
}

} // hid namespace

} // nextfloor namespace