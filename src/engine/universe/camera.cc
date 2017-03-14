/*
* Camera class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/camera.h"

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "engine/universe/room.h"
#include "engine/helpers/proxy_config.h"
#include "engine/helpers/proxygl.h"

namespace engine {

namespace helpers {
namespace proxygl {
    extern GLFWwindow* kGLWindow;
}//namespace proxygl
}//namespace helpers

namespace universe {

namespace {

/* Start record camera position after 10 drawing scene */
static int skip_time = 0;
/* Record mouse wheel scroll */
static float scroll_y = 0.0f;

const std::vector<glm::vec3> coords = {
    /* Back */
    {-1.0f,  2.5f,  1.0f},
    { 1.0f,  2.5f,  1.0f},
    { 1.0f, -2.5f,  1.0f},
    {-1.0f, -2.5f,  1.0f},
    /* Front */
    {-1.0f,  2.5f, -1.0f},
    { 1.0f,  2.5f, -1.0f},
    { 1.0f, -2.5f, -1.0f},
    {-1.0f, -2.5f, -1.0f},
    /* Left */
    {-1.0f,  2.5f,  1.0f},
    {-1.0f,  2.5f, -1.0f},
    {-1.0f, -2.5f, -1.0f},
    {-1.0f, -2.5f,  1.0f},
    /* Right */
    { 1.0f,  2.5f,  1.0f},
    { 1.0f,  2.5f, -1.0f},
    { 1.0f, -2.5f, -1.0f},
    { 1.0f, -2.5f,  1.0f},
    /* Top */
    {-1.0f,  2.5f,  1.0f},
    {-1.0f,  2.5f, -1.0f},
    { 1.0f,  2.5f, -1.0f},
    { 1.0f,  2.5f,  1.0f},
    /* Roof */
    {-1.0f, -2.5f,  1.0f},
    {-1.0f, -2.5f, -1.0f},
    { 1.0f, -2.5f, -1.0f},
    { 1.0f, -2.5f,  1.0f},
};

double last_time = 0;

/*  OnScroll - callback function who record wheel change */
void OnScroll(GLFWwindow* window, double delta_x, double delta_y)
{
    scroll_y += delta_y;
}

}//namespace

/* Constructors */
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
    border_ = engine::geometry::Box(1.0f,
                glm::vec4(cx, cy, cz, 1.0f),
                glm::vec4(0.0f),
                coords);

    type_ = kMODEL3D_CAMERA;
    is_controlled_ = true;
}

/*  Move() - Compute Camera move */
void Camera::Move()
{
    using engine::helpers::proxygl::kGLWindow;
    /* width and height config values */
    using engine::helpers::ProxyConfig;
    float window_width = ProxyConfig::getSetting<float>("width");
    float window_height = ProxyConfig::getSetting<float>("width");

    const float zoom_sensitivity = -0.2f;
    float speed = 4.0f;
    const float mouse_speed = 0.1f;

    double current_time = glfwGetTime();
    float delta_time = float(current_time - last_time);

    /* Get mouse position */
    double xpos = 0, ypos = 0;

    last_time = current_time;
    /* Ensure cursor is well centered before record move */
    if (skip_time++ < 10) {
        glfwSetCursorPos(kGLWindow, window_width/2 , window_height/2);
        return;
    }

    glfwGetCursorPos(kGLWindow, &xpos, &ypos);

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
    border_.set_move(glm::vec3(0.0f));

    /* If shift is pressed => run */
    if (glfwGetKey(kGLWindow, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
        speed *= 2.5f;

    /* Move forward */
    if (glfwGetKey(kGLWindow, GLFW_KEY_UP) == GLFW_PRESS){
        border_.set_move(direction_ * delta_time * speed);
    }
    /* Move backward */
    if (glfwGetKey(kGLWindow, GLFW_KEY_DOWN) == GLFW_PRESS){
        border_.set_move(-direction_ * delta_time * speed);
    }
    /* Strafe right */
    if (glfwGetKey(kGLWindow, GLFW_KEY_RIGHT) == GLFW_PRESS){
        border_.set_move(right * delta_time * speed);
    }
    /* Strafe left */
    if (glfwGetKey(kGLWindow, GLFW_KEY_LEFT) == GLFW_PRESS){
        border_.set_move(-right * delta_time * speed);
    }

    /* Reset Cursor position at center of screen */
    glfwSetCursorPos(kGLWindow, window_width/2, window_height/2);

    /* Manage Field of View with mouse wheel */
    glfwSetScrollCallback(kGLWindow, OnScroll);
    fov_ = fov_ + zoom_sensitivity * scroll_y;
    fov_ = fov_ < 5.0f ? 5.0f : fov_;
    fov_ = fov_ > 130.0f ? 130.0f : fov_;
    scroll_y = 0.0f;
}

/* Test if camera is inside Room */
bool Camera::IsInRoom (const Room &room) const
{
    glm::vec3 location_camera = location();
    glm::vec4 location_room = room.location();


    if (location_camera[0] < location_room[0] + 15.0f
        && location_camera[0] > location_room[0] - 15.0f
        && location_camera[1] < location_room[1] + 4.0f
        && location_camera[1] > location_room[1] - 4.0f
        && location_camera[2] < location_room[2] + 15.0f
        && location_camera[2] > location_room[2] - 15.0f)
            return true;

    return false;
}

}//namespace universe
}//namespace engine
