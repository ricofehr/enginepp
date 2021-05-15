/**
 *  @file mesh.h
 *  @brief Mesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/mesh/mesh.h"


#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include <tbb/mutex.h>

#include "nextfloor/mesh/border.h"
#include "nextfloor/mesh/grid_box.h"

namespace nextfloor {

namespace mesh {

namespace {
/* Unique id for object */
static int sObjectId = 1;
}  // anonymous namespace


std::vector<Mesh*> Mesh::GetMovingObjects() {
    if (border_ != nullptr && border_->IsMoved()) {
        std::vector<Mesh*> leaf(1);
        leaf[0] = this;
        return leaf;
    }
    return std::vector<Mesh*>(0);
}

glm::vec3 Mesh::location() const {
    if (border_ == nullptr) {
        return glm::vec3(0.0f);
    }
    assert(border_ != nullptr);
    return border_->location();
}

glm::vec3 Mesh::dimension() const {
    assert(border_ != nullptr);
    return border_->dimension();
}

float Mesh::diagonal() const {
    assert(border_ != nullptr);
    return border_->diagonal();
}

Border* Mesh::border() const {
    assert(border_ != nullptr);
    return border_.get();
}

void Mesh::set_gridcoords(std::vector<GridBox*> coords_list) { coords_list_ = coords_list; }

void Mesh::ClearCoords()
{
    for (auto& box : coords_list_) {
        box->remove(this);
        delete_gridcoord(box);
    }
}

std::vector<glm::ivec3> Mesh::coords() const
{
    std::vector<glm::ivec3> grid_coords(0);
    for (auto& box : coords_list_) {
        grid_coords.push_back(box->coords());
    }

    return grid_coords;
}

void Mesh::delete_gridcoord(GridBox* grid_box)
{
    for (int cnt = 0; cnt < coords_list_.size(); cnt++) {
        if (coords_list_[cnt] == grid_box) {
            coords_list_.erase(coords_list_.begin() + cnt);
            return;
        }
    }
}

std::vector<Mesh*> Mesh::leafs() {
    std::vector<Mesh*> leaf(1);
    leaf[0] = this;
    return leaf;
}

Mesh::Mesh() {
    id_ = sObjectId++;
}


}  // namespace mesh

}  // namespace nextfloor
