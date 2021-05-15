/**
 *  @file composite_mesh.cc
 *  @brief CompositeMesh class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/mesh/composite_mesh.h"

#include <vector>
#include <cassert>
#include <utility>
#include <tbb/tbb.h>

#include "nextfloor/mesh/mesh.h"

namespace nextfloor {

namespace mesh {

std::vector<Mesh*> CompositeMesh::childs() const
{
    std::vector<Mesh*> ret_childs(0);
    for (auto& object : objects_) {
        ret_childs.push_back(object.get());
    }
    return ret_childs;
}

std::vector<Mesh*> CompositeMesh::leafs()
{
    std::vector<Mesh*> ret_childs(0);
    for (auto& object : objects_) {
        std::vector<Mesh*> grant_childs = object->leafs();
        ret_childs.insert(ret_childs.end(), grant_childs.begin(), grant_childs.end());
    }
    return ret_childs;
}

Mesh* CompositeMesh::add_child(std::unique_ptr<Mesh> object)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    auto object_raw = object.get();
    object_raw->set_parent(this);

    auto initial_objects_size = objects_.size();
    /* Insert Camera as first element. Push on the stack for others */
    if (object_raw->IsCamera()) {
        objects_.insert(objects_.begin(), std::move(object));
    }
    else {
        objects_.push_back(std::move(object));
    }

    /* Ensure object is well added */
    assert(objects_.size() == initial_objects_size + 1);

    return object_raw;
}

std::unique_ptr<Mesh> CompositeMesh::remove_child(Mesh* child)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

    std::unique_ptr<Mesh> ret{nullptr};

    if (child->hasChilds()) {
        std::vector<Mesh*> childs = child->childs();
        for (auto& grant_child : childs) {
            child->remove_child(grant_child);
        }
    }

    for (auto cnt = 0; cnt < objects_.size(); cnt++) {
        if (objects_[cnt].get() == child) {
            auto initial_count_childs = objects_.size();
            ret = std::move(objects_[cnt]);
            objects_.erase(objects_.begin() + cnt);
            /* Ensure child is erased from current objects_ array */
            assert(initial_count_childs == objects_.size() + 1);

            return ret;
        }
    }

    return ret;
}

void CompositeMesh::PrepareDraw(const glm::mat4& view_projection_matrix)
{
    tbb::parallel_for(0, (int)objects_.size(), 1, [&](int i) { objects_[i]->PrepareDraw(view_projection_matrix); });
}


}  // namespace mesh

}  // namespace nextfloor
