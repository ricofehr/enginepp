/**
 *  @file main_menu.cc
 *  @brief MainMenu class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/menu/main_menu.h"

#include <GLFW/glfw3.h>

namespace nextfloor {

namespace menu {


void MainMenu::Init(void* glfw_window)
{
    glfw_window_ = (GLFWwindow*) glfw_window;
}

void MainMenu::Disable()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(context_);
    context_ = nullptr;
    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    is_exit_game_pressed_ = false;
    is_resume_game_pressed_ = false;
}

void MainMenu::Enable()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    context_ = ImGui::CreateContext();
    ImGui::SetCurrentContext(context_);
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(glfw_window_, false);
    ImGui_ImplOpenGL3_Init("#version 150");

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    is_exit_game_pressed_ = false;
    is_resume_game_pressed_ = false;
}

void MainMenu::MenuLoop()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Main Menu");
    if (ImGui::Button("Resume Game")) {
        is_resume_game_pressed_ = true;
    }

    if (ImGui::Button("Exit Game")) {
        is_exit_game_pressed_ = true;
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

} // namespace menu

} // namespace nextfloor
