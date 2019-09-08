/**
 *  @file scene_input.h
 *  @brief SceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SCENEINPUT_H_
#define NEXTFLOOR_RENDERER_SCENEINPUT_H_

#include <functional>
#include <glm/glm.hpp>

#include "nextfloor/renderer/scene_window.h"

namespace nextfloor {

namespace renderer {

class SceneInput {

public:
    /**
     *  Action Button
     */
    static constexpr int kINPUT_UP = 0;
    static constexpr int kINPUT_DOWN = 1;
    static constexpr int kINPUT_LEFT = 2;
    static constexpr int kINPUT_RIGHT = 3;
    static constexpr int kINPUT_JUMP = 4;
    static constexpr int kINPUT_RUN = 5;
    static constexpr int kINPUT_FIRE = 6;

    static constexpr int kACTIONS = 7;

    virtual ~SceneInput() = default;

    virtual void PollEvents() = 0;
    virtual bool IsCloseWindowEventOccurs() = 0;
    virtual bool IsPressed(int ACTION_BUTTON) = 0;
    virtual glm::vec2 GetCursorPos() = 0;
    virtual void SetCursorPos(float x, float y) = 0;
    // virtual void SetScrollCallBack(void (*on_scroll)(void* window, double delta_x, double delta_y)) = 0;

protected:
    SceneInput() = default;

    SceneInput(SceneInput&&) = default;
    SceneInput& operator=(SceneInput&&) = default;
    SceneInput(const SceneInput&) = default;
    SceneInput& operator=(const SceneInput&) = default;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_SCENEINPUT_H_
