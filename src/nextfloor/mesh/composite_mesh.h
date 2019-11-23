/**
 *  @file composite_mesh.h
 *  @brief CompositeMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_COMPOSITEMESH_H_
#define NEXTFLOOR_MESH_COMPOSITEMESH_H_

#include "nextfloor/mesh/default_mesh.h"

#include <memory>
#include <tbb/mutex.h>
#include <glm/glm.hpp>

namespace nextfloor {

namespace mesh {

/**
 *  @class CompositeMesh
 *  @brief Abstract class who defines composite pattern for meshes
 */
class CompositeMesh : public DefaultMesh {

public:
    ~CompositeMesh() override = default;

    friend bool operator==(const CompositeMesh& o1, const CompositeMesh& o2);
    friend bool operator!=(const CompositeMesh& o1, const CompositeMesh& o2);

    Mesh* add_child(std::unique_ptr<Mesh> object) override;
    std::unique_ptr<Mesh> remove_child(Mesh* child) override;

    int id() const final { return id_; }
    void set_parent(Mesh* parent) final { parent_ = parent; }
    bool hasChilds() const final { return objects_.size() != 0; }
    bool hasNoChilds() const final { return objects_.size() == 0; }

    std::vector<Mesh*> AllStubMeshs() final;
    std::vector<Mesh*> childs() const final;
    std::vector<Mesh*> descendants() const final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) override;

    std::string class_name() override { return "CompositeMesh"; }

protected:
    CompositeMesh();

    CompositeMesh(CompositeMesh&&) = delete;
    CompositeMesh& operator=(CompositeMesh&&) = delete;
    CompositeMesh(const CompositeMesh&) = delete;
    CompositeMesh& operator=(const CompositeMesh&) = delete;

    Mesh* parent_{nullptr};
    std::vector<std::unique_ptr<Mesh>> objects_;
    tbb::mutex mutex_;

private:
    int id_{0};
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_COMPOSITEMESH_H_
