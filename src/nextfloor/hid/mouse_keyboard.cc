/**
 *  @file keyboard_mouse.cc
 *  @brief KeyboardMouse Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/mouse_keyboard.h"

#include <cassert>

#include "nextfloor/core/common_services.h"

namespace nextfloor {

namespace hid {

namespace {

/* Start record camera position after 10 drawing scene */
static int sSkipTime = 0;

}  // namespace

MouseKeyboard::MouseKeyboard(std::unique_ptr<nextfloor::gameplay::SceneInput> scene_input)
{
    scene_input_ = std::move(scene_input);
}

bool MouseKeyboard::isPressed(int action_button)
{
    return scene_input_->IsPressed(action_button);
}

nextfloor::gameplay::HIDPointer MouseKeyboard::RecordHIDPointer(double elapsed_time)
{
    using nextfloor::core::CommonServices;
    float window_width = CommonServices::getConfig()->getWindowWidth();
    float window_height = CommonServices::getConfig()->getWindowHeight();

    /* Get mouse position */
    glm::vec2 mouse_position;

    /* Ensure cursor is well centered before record move */
    if (sSkipTime < kFramesToBeReady) {
        assert(scene_input_ != nullptr);
        scene_input_->SetCursorPos(window_width / 2, window_height / 2);
        mouse_position.x = window_width / 2;
        mouse_position.y = window_height / 2;
        sSkipTime++;
    }
    else {
        mouse_position = scene_input_->GetCursorPos();
    }


    nextfloor::gameplay::HIDPointer pointer
      = {kHidSpeed * elapsed_time * static_cast<float>(window_width / 2 - mouse_position.x),
         kHidSpeed * elapsed_time * static_cast<float>(window_height / 2 - mouse_position.y)};

    /* Reset Cursor position at center of screen */
    scene_input_->SetCursorPos(window_width / 2, window_height / 2);

    return pointer;
}

void MouseKeyboard::PollEvents()
{
    scene_input_->PollEvents();
}

bool MouseKeyboard::IsCloseWindowEventOccurs()
{
    return scene_input_->IsCloseWindowEventOccurs();
}

}  // namespace hid

}  // namespace nextfloor
