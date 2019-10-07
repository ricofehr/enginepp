/**
 *  @file mouse_keyboard.h
 *  @brief MouseKeyboard Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_MOUSEKEYBOARD_H_
#define NEXTFLOOR_HID_MOUSEKEYBOARD_H_

#include "nextfloor/gameplay/hid.h"

#include <memory>

#include "nextfloor/gameplay/scene_input.h"

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
class MouseKeyboard : public nextfloor::gameplay::HID {

public:
    /**
     *  Default constructor
     */
    MouseKeyboard(std::unique_ptr<nextfloor::gameplay::SceneInput> scene_input);
    ~MouseKeyboard() final = default;

    MouseKeyboard(MouseKeyboard&&) = default;
    MouseKeyboard& operator=(MouseKeyboard&&) = default;
    MouseKeyboard(const MouseKeyboard&) = delete;
    MouseKeyboard& operator=(const MouseKeyboard&) = delete;


    /**
     *  Check if a button is pressed
     *  @param  window        GL Main Window
     *  @param  ACTION_BUTTON Button to be checked
     *  @return               True if pressed
     */
    bool isPressed(int ACTION_BUTTON) final;

    /**
     * Get HID Pointer angles changes
     * @param  window GL Window
     * @return        HIDPointer struct with orientation angles changes
     */
    nextfloor::gameplay::HIDPointer RecordHIDPointer(double elapsed_time) final;

    /**
     *  Return current "Field of View"
     *  @param  window GL Main Window
     *  @return        Current Field of View
     */
    float RecordFOV() final;

    virtual void PollEvents() final;

    virtual bool IsCloseWindowEventOccurs() final;

private:
    std::unique_ptr<nextfloor::gameplay::SceneInput> scene_input_;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_MOUSEKEYBOARD_H_