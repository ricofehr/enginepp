/**
 *  @file scene_input.h
 *  @brief SceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_SCENEINPUT_H_
#define NEXTFLOOR_GAMEPLAY_SCENEINPUT_H_

#include <glm/glm.hpp>

namespace nextfloor {

namespace gameplay {

class SceneInput {

public:
    /**
     *  Action Button
     */
    static constexpr int kInputUp = 0;
    static constexpr int kInputDown = 1;
    static constexpr int kInputLeft = 2;
    static constexpr int kInputRight = 3;
    static constexpr int kInputSpace = 4;

    virtual ~SceneInput() = default;

    virtual void PollEvents() = 0;
    virtual bool IsCloseWindowEventOccurs() = 0;
    virtual bool IsPressed(int action_button) = 0;
    virtual glm::vec2 GetCursorPos() = 0;
    virtual void SetCursorPos(float x, float y) = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_SCENEINPUT_H_
