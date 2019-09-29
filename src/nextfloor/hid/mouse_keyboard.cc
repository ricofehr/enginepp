/**
 *  @file keyboard_mouse.cc
 *  @brief KeyboardMouse Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/hid/mouse_keyboard.h"

#include <functional>

#include "nextfloor/core/common_services.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace hid
 *  @brief hid namespace
 */
namespace hid {

namespace {

/* Start record camera position after 10 drawing scene */
static int sSkipTime = 0;

/* Record mouse wheel scroll */
static float sScrollY = 0.0f;


/*
 *  OnScroll - callback function who record wheel change
 */
static void OnScroll(void* window, double delta_x, double delta_y)
{
    sScrollY += delta_y;
}

}  // namespace

MouseKeyboard::MouseKeyboard()
{
    auto factory = nextfloor::core::CommonServices::getFactory();
    scene_input_ = factory->MakeSceneInput();
}

/**
 *  Check if a button is pressed
 *  @param  window        GL Main Window
 *  @param  ACTION_BUTTON Button to be checked
 *  @return               True if pressed
 */
bool MouseKeyboard::isPressed(int ACTION_BUTTON)
{
    return scene_input_->IsPressed(ACTION_BUTTON);
}


/**
 * Get HID Pointer angles changes
 * @param  window GL Window
 * @return        HIDPointer struct with orientation angles changes
 */
HIDPointer MouseKeyboard::RecordHIDPointer(double elapsed_time)
{
    using nextfloor::core::CommonServices;
    float window_width = CommonServices::getConfig()->getWindowWidth();
    float window_height = CommonServices::getConfig()->getWindowHeight();

    const float hid_speed = 0.1f;

    /* Get mouse position */
    glm::vec2 mouse_position;

    /* Ensure cursor is well centered before record move */
    if (sSkipTime++ < 10) {
        scene_input_->SetCursorPos(window_width / 2, window_height / 2);
        mouse_position.x = window_width / 2;
        mouse_position.y = window_height / 2;
    }
    else {
        mouse_position = scene_input_->GetCursorPos();
    }


    HIDPointer pointer = {hid_speed * elapsed_time * static_cast<float>(window_width / 2 - mouse_position.x),
                          hid_speed * elapsed_time * static_cast<float>(window_height / 2 - mouse_position.y)};

    /* Reset Cursor position at center of screen */
    scene_input_->SetCursorPos(window_width / 2, window_height / 2);

    return pointer;
}

/**
 *  Return current "Field of View"
 *  @param  window GL Main Window
 *  @return        Current Field of View
 */
float MouseKeyboard::RecordFOV()
{
    const float zoom_sensitivity = -0.2f;
    float delta_fov = 0.0f;

    /* Manage Field of View with mouse wheel */
    // scene_input_->SetScrollCallBack(std::function<void(void*, double, double)>(OnScroll));
    delta_fov = zoom_sensitivity * sScrollY;
    sScrollY = 0.0f;

    return delta_fov;
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
