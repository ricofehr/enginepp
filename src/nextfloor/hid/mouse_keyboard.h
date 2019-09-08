/**
 *  @file mouse_keyboard.h
 *  @brief MouseKeyboard Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_MOUSEKEYBOARD_H_
#define NEXTFLOOR_HID_MOUSEKEYBOARD_H_

#include "nextfloor/hid/hid.h"

#include <memory>

#include "nextfloor/renderer/scene_input.h"

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

/**
 *  Manages keyboard / Mouse Controllers
 *  Implements HID Interface
 */
class MouseKeyboard : public HID {

public:
    /**
     *  Default constructor
     */
    MouseKeyboard();

    MouseKeyboard(MouseKeyboard&&) = default;
    MouseKeyboard& operator=(MouseKeyboard&&) = default;
    MouseKeyboard(const MouseKeyboard&) = default;
    MouseKeyboard& operator=(const MouseKeyboard&) = default;

    virtual ~MouseKeyboard() override = default;

    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    bool isPressed(int ACTION_BUTTON) override final;

    /**
     * Get HID Pointer angles changes
     * @param  window GL Window
     * @return        HIDPointer struct with orientation angles changes
     */
    HIDPointer RecordHIDPointer() override final;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    float RecordFOV() override final;

    virtual void PollEvents() override final;

    virtual bool IsCloseWindowEventOccurs() override final;

private:
    std::unique_ptr<nextfloor::renderer::SceneInput> scene_input_;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_MOUSEKEYBOARD_H_