/*
* Model3d class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/model3d.h"

#include <SOIL/SOIL.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cilk/cilk.h>

#include "engine/physics/cl_collision_engine.h"
#include "engine/physics/serial_collision_engine.h"
#include "engine/physics/cilk_collision_engine.h"
#include "engine/core/config_engine.h"

namespace engine {
namespace universe {

//std::unique_ptr<engine::physics::CollisionEngine> Model3D::collision_engine_ = nullptr;

namespace {
    /* Unique id for object */
    static int sObjectId = 1;
}

/* Constructor */
Model3D::Model3D()
{
    id_ = sObjectId++;
    distance_ = -1.0f;
    obstacle_ = nullptr;
    id_last_collision_ = 0;
    is_controlled_ = false;
    is_crossed_ = false;
    InitCollisionEngine();
}

/* Comparaison operator */
bool operator==(const Model3D& o1, const Model3D& o2)
{
    return o1.id() == o2.id();
}

bool operator!=(const Model3D& o1, const Model3D& o2)
{
    return o1.id() != o2.id();
}

/* Init CollisionEngine pointer */
void Model3D::InitCollisionEngine()
{
    using engine::physics::CollisionEngine;
    using engine::core::ConfigEngine;

    /* Get parallell type from config */
    int type_parallell = ConfigEngine::getSetting<int>("parallell");

    switch (type_parallell) {
        case CollisionEngine::kPARALLELL_CILK:
            collision_engine_ = engine::physics::CilkCollisionEngine::Instance();
            break;
        case CollisionEngine::kPARALLELL_CL:
            collision_engine_ = engine::physics::CLCollisionEngine::Instance();
            break;
        default:
            collision_engine_ = engine::physics::SerialCollisionEngine::Instance();
            break;
    }
}

/* Compute coords and move for the model */
void Model3D::PrepareDraw(Camera *cam)
{
    border_.set_distance(distance_);
    border_.MoveCoords();
    for (auto &element : elements_) {
        element->set_distance(distance_);
        element->ComputeMVP(cam);
    }
    distance_ = -1.0f;
    /* An object cant touch same object twice, except camera */
    id_last_collision_ = -1;
    if (!is_controlled_ && obstacle_ != nullptr) {
        id_last_collision_ = obstacle_->id();
    }
    obstacle_ = nullptr;
}

/* Draw the model */
void Model3D::Draw()
{
    for (auto &element : elements_) {
        element->Draw();
    }
}

}//namespace geometry
}//namespace engine
