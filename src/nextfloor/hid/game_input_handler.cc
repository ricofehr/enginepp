/**
 *  @file input_handler.cc
 *  @brief InputHandler Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/game_input_handler.h"

#include <memory>

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/gameplay/scene_input.h"

namespace nextfloor {

namespace hid {

/**
 *  Constructor
 *  @param  window  GL Main Window
 */
GameInputHandler::GameInputHandler(std::unique_ptr<nextfloor::gameplay::HID> hid)
{
    hid_ = std::move(hid);
}

void GameInputHandler::InitCommands(const nextfloor::gameplay::ActionFactory& action_factory)
{
    /* Init Command Pointers */
    move_up_command_ = action_factory.MakeMoveUpAction();
    move_down_command_ = action_factory.MakeMoveDownAction();
    move_left_command_ = action_factory.MakeMoveLeftAction();
    move_right_command_ = action_factory.MakeMoveRightAction();
    jump_command_ = action_factory.MakeJumpAction();
    run_command_ = action_factory.MakeRunAction();
    fire_command_ = action_factory.MakeFireAction();
}

nextfloor::gameplay::Action* GameInputHandler::HandlerInput()
{
    using nextfloor::gameplay::SceneInput;

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