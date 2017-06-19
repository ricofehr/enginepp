/**
 *  @file camera.cc
 *  @brief Camera class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "engine/universe/camera.h"

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "engine/core/config_engine.h"
#include "engine/renderer/loopgl.h"

namespace engine {

namespace universe {

namespace {

/* Start record camera position after 10 drawing scene */
static int sSkipTime = 0;

/* Record mouse wheel scroll */
static float sScrollY = 0.0f;

/* Camera coords, used for collision computes */
static const std::vector<glm::vec3> sCameraCoords = {
    /* Back */
    {-0.25f,  0.5f,  0.25f},
    { 0.25f,  0.5f,  0.25f},
    { 0.25f, -0.5f,  0.25f},
    {-0.25f, -0.5f,  0.25f},
    /* Front */
    {-0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    {-0.25f, -0.5f, -0.25f},
    /* Left */
    {-0.25f,  0.5f,  0.25f},
    {-0.25f,  0.5f, -0.25f},
    {-0.25f, -0.5f, -0.25f},
    {-0.25f, -0.5f,  0.25f},
    /* Right */
    { 0.25f,  0.5f,  0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f,  0.25f},
    /* Top */
    {-0.25f,  0.5f,  0.25f},
    {-0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f, -0.25f},
    { 0.25f,  0.5f,  0.25f},
    /* Roof */
    {-0.25f, -0.5f,  0.25f},
    {-0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f, -0.25f},
    { 0.25f, -0.5f,  0.25f},
};

/* Record a time */
static double sLastTime = 0;

/*
 *  OnScroll - callback function who record wheel change
 */
static void OnScroll(GLFWwindow* window, double delta_x, double delta_y)
{
    sScrollY += delta_y;
}

} // anonymous namespace

/* Define global active camera */
Camera* Camera::active_ = nullptr;

Camera::Camera()
       :Camera(15.0f, 100.0f, 15.0f, 15.0f, 0.0f, 15.0f, 0.0f, 0.0f, 1.0f) {}

Camera::Camera(float cx, float cy, float cz,
               float vx, float vy, float vz,
               float hx, float hy, float hz)
{
    direction_ = glm::vec3(vx, vy, vz);
    head_ = glm::vec3(hx, hy, hz);
    horizontal_angle_ = 3.14f;
    vertical_angle_ = 0.0;
    fov_ = 45.0f;

    using engine::graphics::Border;
    border_ = std::make_unique<Border>(1.0f,
                glm::vec4(cx, cy, cz, 1.0f),
                glm::vec4(0.0f),
                sCameraCoords);

    type_ = kMODEL3D_CAMERA;

    if (active_ == nullptr) {
        set_active();
    }
}

/*  Move() - Compute Camera move */
void Camera::RecordHID() noexcept
{
    using engine::renderer::LoopGL;

    /* width and height config values */
    using engine::core::ConfigEngine;
    float window_width = ConfigEngine::getSetting<float>("width");
    float window_height = ConfigEngine::getSetting<float>("width");

    /* Dont apply movefactor to camera */
    using engine::graphics::Shape3D;
    float speed = 3.5f / Shape3D::sMoveFactor;

    const float zoom_sensitivity = -0.2f;
    const float mouse_speed = 0.1f;

    double current_time = glfwGetTime();
    float delta_time = float(current_time - sLastTime);

    /* Get mouse position */
    double xpos = 0, ypos = 0;

    sLastTime = current_time;
    /* Ensure cursor is well centered before record move */
    if (sSkipTime++ < 10) {
        glfwSetCursorPos(LoopGL::sGLWindow, window_width/2 , window_height/2);
        return;
    }

    glfwGetCursorPos(LoopGL::sGLWindow, &xpos, &ypos);

    /* Compute new orientation */
    horizontal_angle_ += mouse_speed * delta_time * static_cast<float>(window_width/2 - xpos);
    vertical_angle_   += mouse_speed * delta_time * static_cast<float>(window_height/2 - ypos);

    /* Direction : Spherical coordinates to Cartesian coordinates conversion */
    direction_[0] = cos(vertical_angle_) * sin(horizontal_angle_);
    direction_[1] = sin(vertical_angle_);
    direction_[2] = cos(vertical_angle_) * cos(horizontal_angle_);

    /* Right vector */
    glm::vec3 right = glm::vec3(
        sin(horizontal_angle_ - 3.14f/2.0f),
        0,
        cos(horizontal_angle_ - 3.14f/2.0f)
    );

    /* head_ vector : perpendicular to both direction and right */
    head_ = glm::cross(right, direction_);

    /* Reset move translate */
    border_->set_move(glm::vec3(0.0f));

    /* If shift is pressed => run */
    if (glfwGetKey(LoopGL::sGLWindow, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        speed *= 2.5f;
    }

    /* Move forward */
    if (glfwGetKey(LoopGL::sGLWindow, GLFW_KEY_UP) == GLFW_PRESS) {
        border_->set_move(direction_ * delta_time * speed);
    }
    /* Move backward */
    if (glfwGetKey(LoopGL::sGLWindow, GLFW_KEY_DOWN) == GLFW_PRESS) {
        border_->set_move(-direction_ * delta_time * speed);
    }
    /* Strafe right */
    if (glfwGetKey(LoopGL::sGLWindow, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        border_->set_move(right * delta_time * speed);
    }
    /* Strafe left */
    if (glfwGetKey(LoopGL::sGLWindow, GLFW_KEY_LEFT) == GLFW_PRESS) {
        border_->set_move(-right * delta_time * speed);
    }

    /* Reset Cursor position at center of screen */
    glfwSetCursorPos(LoopGL::sGLWindow, window_width/2, window_height/2);

    /* Manage Field of View with mouse wheel */
    glfwSetScrollCallback(LoopGL::sGLWindow, OnScroll);
    fov_ = fov_ + zoom_sensitivity * sScrollY;
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
    sScrollY = 0.0f;
}

} // namespace universe

} // namespace engine
