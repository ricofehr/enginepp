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
}

nextfloor::gameplay::Action* GameInputHandler::HandlerInput()
{
    using nextfloor::gameplay::SceneInput;

    if (hid_->isPressed(SceneInput::kInputUp)) {
        return move_up_command_.get();
    }

    if (hid_->isPressed(SceneInput::kInputDown)) {
        return move_down_command_.get();
    }

    if (hid_->isPressed(SceneInput::kInputLeft)) {
        return move_left_command_.get();
    }

    if (hid_->isPressed(SceneInput::kInputRight)) {
        return move_right_command_.get();
    }

    return nullptr;
}

nextfloor::gameplay::HIDPointer GameInputHandler::RecordHIDPointer(double elapsed_time)
{
    return hid_->RecordHIDPointer(elapsed_time);
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