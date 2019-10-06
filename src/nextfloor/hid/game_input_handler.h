/**
 *  @file game_input_handler.h
 *  @brief GameInputHandler Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_GAMEINPUTHANDLER_H_
#define NEXTFLOOR_HID_GAMEINPUTHANDLER_H_

#include "nextfloor/gameplay/input_handler.h"

#include "nextfloor/gameplay/action_factory.h"
#include "nextfloor/gameplay/hid_factory.h"
#include "nextfloor/gameplay/renderer_factory.h"

namespace nextfloor {

namespace hid {

/**
 *  @class GameInputHandler
 *  @brief GameInputHandler maintains input commands from engine
 */
class GameInputHandler : public nextfloor::gameplay::InputHandler {

public:
    GameInputHandler(const nextfloor::gameplay::HidFactory& hid_factory,
                     const nextfloor::gameplay::ActionFactory& factory,
                     nextfloor::gameplay::RendererFactory* renderer_factory);
    ~GameInputHandler() final = default;

    GameInputHandler(GameInputHandler&&) = default;
    GameInputHandler& operator=(GameInputHandler&&) = default;
    GameInputHandler(const GameInputHandler&) = delete;
    GameInputHandler& operator=(const GameInputHandler&) = delete;

    /**
     *  Get Current State Input
     *  @return Command Object
     */
    nextfloor::gameplay::Action* HandlerInput() final;

    /**
     *  Get HID Pointer angles changes
     *  @return HIDPointer struct
     */
    nextfloor::gameplay::HIDPointer RecordHIDPointer(double elapsed_time) final;

    /**
     *  Get FOV (FieldOfView) Current State
     *  @return FOV value
     */
    float RecordFOV() final;

    void PollEvents() final;
    bool IsCloseWindowEventOccurs() final;

private:
    /** The main hid */
    std::unique_ptr<nextfloor::gameplay::HID> hid_{nullptr};

    /** Character Commands */
    std::unique_ptr<nextfloor::gameplay::Action> move_up_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> move_down_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> move_left_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> move_right_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> jump_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> run_command_{nullptr};
    std::unique_ptr<nextfloor::gameplay::Action> fire_command_{nullptr};
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_INPUTHANDLER_H_
