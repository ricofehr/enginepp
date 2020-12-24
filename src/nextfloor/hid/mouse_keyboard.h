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

namespace nextfloor {

namespace hid {

/**
 *  Manages keyboard / Mouse Controllers
 *  Implements HID Interface
 */
class MouseKeyboard : public nextfloor::gameplay::HID {

public:
    MouseKeyboard(std::unique_ptr<nextfloor::gameplay::SceneInput> scene_input);
    ~MouseKeyboard() final = default;

    MouseKeyboard(MouseKeyboard&&) = default;
    MouseKeyboard& operator=(MouseKeyboard&&) = default;
    MouseKeyboard(const MouseKeyboard&) = delete;
    MouseKeyboard& operator=(const MouseKeyboard&) = delete;

    bool isPressed(int action_button) final;
    bool isReleased(int action_button) final;

    /**
     * Get HID Pointer angles changes
     * @param  elapsed_time since last record
     * @return        HIDPointer struct with orientation angles changes
     */
    nextfloor::gameplay::HIDPointer RecordHIDPointer(double elapsed_time) final;

    void PollEvents() final;
    void ResetPointer() final;

private:
    static constexpr float kHidSpeed = 0.1f;
    static constexpr int kFramesToBeReady = 10;

    std::unique_ptr<nextfloor::gameplay::SceneInput> scene_input_{nullptr};
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_MOUSEKEYBOARD_H_