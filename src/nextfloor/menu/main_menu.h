/**
 *  @file main_menu.h
 *  @brief Main Menu class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MENU_MAINMENU_H_
#define NEXTFLOOR_MENU_MAINMENU_H_

#include "nextfloor/gameplay/menu.h"

#include <memory>

#include "ext/imgui/imgui.h"
#include "ext/imgui/imgui_impl_glfw.h"
#include "ext/imgui/imgui_impl_opengl3.h"

namespace nextfloor {

namespace menu {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class MainMenu : public nextfloor::gameplay::Menu {

public:
    MainMenu() = default;
    ~MainMenu() final = default;

    MainMenu(const MainMenu&) = delete;
    MainMenu& operator=(const MainMenu&) = delete;

    void Enable() final;
    void Disable() final;
    void MenuLoop() final;
    void Init(void* glfw_window) final;
    bool IsResumeGamePressed() final { return is_resume_game_pressed_; }
    bool IsExitGamePressed() final { return is_exit_game_pressed_; }

private:
    GLFWwindow *glfw_window_{nullptr};
    ImGuiContext *context_{nullptr};
    bool is_resume_game_pressed_{false};
    bool is_exit_game_pressed_{false};
};

}  // namespace menu

}  // namespace nextfloor

#endif  // NEXTFLOOR_MENU_MAINMENU_H_
