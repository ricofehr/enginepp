/**
 *  @file composite_mesh.h
 *  @brief CompositeMesh class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_COMPOSITEMESH_H_
#define NEXTFLOOR_MESH_COMPOSITEMESH_H_

#include "nextfloor/mesh/mesh.h"

#include <memory>
#include <vector>


namespace nextfloor {

namespace mesh {

/**
 *  @class CompositeMesh
 *  @brief Abstract class who defines composite pattern for meshes
 */
class CompositeMesh : public Mesh {

public:
    ~CompositeMesh() override = default;

    Mesh* add_child(std::unique_ptr<Mesh> object) override;
    std::unique_ptr<Mesh> remove_child(Mesh* child) override;

    bool hasChilds() const final { return objects_.size() != 0; }
    bool hasNoChilds() const final { return objects_.size() == 0; }

    std::vector<Mesh*> leafs() final;
    std::vector<Mesh*> childs() const final;
    void PrepareDraw(const glm::mat4& view_projection_matrix) override;

    std::string class_name() const override { return "CompositeMesh"; }

protected:
    CompositeMesh() = default;

    CompositeMesh(CompositeMesh&&) = delete;
    CompositeMesh& operator=(CompositeMesh&&) = delete;
    CompositeMesh(const CompositeMesh&) = delete;
    CompositeMesh& operator=(const CompositeMesh&) = delete;

    std::vector<std::unique_ptr<Mesh>> objects_;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_COMPOSITEMESH_H_
