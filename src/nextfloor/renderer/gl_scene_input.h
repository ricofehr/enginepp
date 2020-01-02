/**
 *  @file gl_scene_input.h
 *  @brief GlSceneInput class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSCENEINPUT_H_
#define NEXTFLOOR_RENDERER_GLSCENEINPUT_H_

#include "nextfloor/gameplay/scene_input.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace nextfloor {

namespace renderer {

class GlSceneInput : public nextfloor::gameplay::SceneInput {

public:
    GlSceneInput(GLFWwindow* glfw_window);
    ~GlSceneInput() final = default;

    void PollEvents() final;
    bool IsCloseWindowEventOccurs() final;
    bool IsPressed(int action_button) final;
    glm::vec2 GetCursorPos() final;
    void SetCursorPos(float x, float y) final;

private:
    int getKeyValueForAction(int action_button);

    GLFWwindow* glfw_window_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSCENEINPUT_H_
