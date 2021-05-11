/**
 *  @file main_menu.cc
 *  @brief MainMenu class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/menu/main_menu.h"

#include <GLFW/glfw3.h>

#include "nextfloor/core/common_services.h"


namespace nextfloor {

namespace menu {


void MainMenu::Init(void* glfw_window)
{
    glfw_window_ = (GLFWwindow*) glfw_window;
    InitOptionValues();
}

void MainMenu::InitOptionValues()
{
    using nextfloor::core::CommonServices;
    is_option_grid_mode_ = CommonServices::getConfig()->isGridMode();
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
    ImGui_ImplOpenGL3_Init("#version 410");

    glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    is_exit_game_pressed_ = false;
    is_resume_game_pressed_ = false;
}

void MainMenu::MenuLoop()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (is_option_display_) {
        OptionList();
    } else {
        MainList();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void MainMenu::MainList()
{
    ImGui::Begin("Main Menu");
    ImGui::SetWindowSize(ImVec2(150, 100));

    if (ImGui::Button("Resume GL Scene")) {
        is_resume_game_pressed_ = true;
    }

    if (ImGui::Button("Options")) {
        is_option_display_ = true;
    }

    if (ImGui::Button("Exit GL Scene")) {
        is_exit_game_pressed_ = true;
    }
    ImGui::End();
}

void MainMenu::OptionList()
{
    ImGui::Begin("Option Menu");
    ImGui::SetWindowSize(ImVec2(150, 80));

    ImGui::Checkbox("Grid Mode", &is_option_grid_mode_);

    if (ImGui::Button("Back")) {
        is_option_display_ = false;
    }
    ImGui::End();

    UpdateConfigValues();
}

void MainMenu::UpdateConfigValues() const
{
    using nextfloor::core::CommonServices;
    CommonServices::getConfig()->setGridMode(is_option_grid_mode_);
}

} // namespace menu

} // namespace nextfloor
