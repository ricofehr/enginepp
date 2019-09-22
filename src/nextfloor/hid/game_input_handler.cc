/**
 *  @file input_handler.cc
 *  @brief InputHandler Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/game_input_handler.h"

#include <memory>

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/renderer/scene_input.h"

#include "nextfloor/actions/moveup_action.h"
#include "nextfloor/actions/movedown_action.h"
#include "nextfloor/actions/moveleft_action.h"
#include "nextfloor/actions/moveright_action.h"
#include "nextfloor/actions/jump_action.h"
#include "nextfloor/actions/run_action.h"
#include "nextfloor/actions/fire_action.h"

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
    using nextfloor::core::CommonServices;
    move_up_command_ = CommonServices::getFactory().MakeMoveUpAction();
    move_down_command_ = CommonServices::getFactory().MakeMoveDownAction();
    move_left_command_ = CommonServices::getFactory().MakeMoveLeftAction();
    move_right_command_ = CommonServices::getFactory().MakeMoveRightAction();
    jump_command_ = CommonServices::getFactory().MakeJumpAction();
    run_command_ = CommonServices::getFactory().MakeRunAction();
    fire_command_ = CommonServices::getFactory().MakeFireAction();
}

nextfloor::actions::Action* GameInputHandler::HandlerInput()
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

HIDPointer GameInputHandler::RecordHIDPointer()
{
    return hid_->RecordHIDPointer();
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