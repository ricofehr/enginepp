/**
 *  @file gl_scene_input.h
 *  @brief GlSceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSCENEINPUT_H_
#define NEXTFLOOR_RENDERER_GLSCENEINPUT_H_

#include "nextfloor/renderer/scene_input.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace nextfloor {

namespace renderer {

class GlSceneInput : public SceneInput {

public:
    GlSceneInput(SceneWindow* window);

    GlSceneInput(GlSceneInput&&) = default;
    GlSceneInput& operator=(GlSceneInput&&) = default;
    GlSceneInput(const GlSceneInput&) = default;
    GlSceneInput& operator=(const GlSceneInput&) = default;

    virtual ~GlSceneInput() override = default;

    virtual void PollEvents() override;
    virtual bool IsCloseWindowEventOccurs() override;
    virtual bool IsPressed(int ACTION_BUTTON) override;
    virtual glm::vec2 GetCursorPos() override;
    virtual void SetCursorPos(float x, float y) override;
    // virtual void SetScrollCallBack(void (*on_scroll)(void* window, double delta_x, double delta_y)) override;

private:
    int getKeyValueForAction(int ACTION_BUTTON);

    GLFWwindow* glfw_window_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSCENEINPUT_H_
