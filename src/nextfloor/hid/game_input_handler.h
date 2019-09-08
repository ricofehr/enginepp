/**
 *  @file game_input_handler.h
 *  @brief GameInputHandler Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_GAMEINPUTHANDLER_H_
#define NEXTFLOOR_HID_GAMEINPUTHANDLER_H_

#include "nextfloor/hid/input_handler.h"

namespace nextfloor {

namespace hid {

/**
 *  @class GameInputHandler
 *  @brief GameInputHandler maintains input commands from engine
 */
class GameInputHandler : public InputHandler {

public:
    GameInputHandler();

    GameInputHandler(GameInputHandler&&) noexcept = default;
    GameInputHandler& operator=(GameInputHandler&&) noexcept = default;
    GameInputHandler(const GameInputHandler&) = default;
    GameInputHandler& operator=(const GameInputHandler&) = default;

    virtual ~GameInputHandler() override = default;

    /**
     *  Get Current State Input
     *  @return Command Object
     */
    virtual nextfloor::actions::Action* HandlerInput() override final;

    /**
     *  Get HID Pointer angles changes
     *  @return HIDPointer struct
     */
    virtual HIDPointer RecordHIDPointer() override final;

    /**
     *  Get FOV (FieldOfView) Current State
     *  @return FOV value
     */
    virtual float RecordFOV() override final;

    virtual void PollEvents() override final;
    virtual bool IsCloseWindowEventOccurs() override final;

private:
    /** The main hid */
    std::unique_ptr<HID> hid_{nullptr};

    /** Character Commands */
    std::unique_ptr<nextfloor::actions::Action> move_up_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> move_down_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> move_left_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> move_right_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> jump_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> run_command_{nullptr};
    std::unique_ptr<nextfloor::actions::Action> fire_command_{nullptr};
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_INPUTHANDLER_H_
