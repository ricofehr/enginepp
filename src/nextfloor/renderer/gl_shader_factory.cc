/**
 *  @file gl_renderer_factory.c
 *  @brief Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/renderer/gl_shader_factory.h"

#include <tbb/mutex.h>
#include <cassert>
#include <sstream>

#include "nextfloor/renderer/vertex_gl_shader.h"
#include "nextfloor/renderer/fragment_gl_shader.h"

namespace nextfloor {

namespace renderer {

namespace {

static bool sInstanciated = false;

static constexpr const char sShaderFolder[] = "glsl";
static constexpr const char sVertexShaderSuffix[] = "VertexShader.vertexshader";
static constexpr const char sFragmentShaderSuffix[] = "FragmentShader.fragmentshader";


std::string getVertexShaderPath(std::string label)
{
    std::stringstream ss_path;
    ss_path << sShaderFolder << "/" << label << sVertexShaderSuffix;
    return ss_path.str();
}
std::string getFragmentShaderPath(std::string label)
{
    std::stringstream ss_path;
    ss_path << sShaderFolder << "/" << label << sFragmentShaderSuffix;
    return ss_path.str();
}

}  // anonymous namespace

GlShaderFactory::GlShaderFactory()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

Shader* GlShaderFactory::MakeVertexShader(const std::string& shader_label, unsigned int program_id)
{
    static tbb::mutex vertex_mutex_;

    std::string shader_path = getVertexShaderPath(shader_label);

    vertex_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        shaders_[shader_path] = std::make_unique<VertexGlShader>(shader_path, program_id);
    }
    vertex_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

Shader* GlShaderFactory::MakeFragmentShader(const std::string& shader_label, unsigned int program_id)
{
    static tbb::mutex fragment_mutex_;

    std::string shader_path = getFragmentShaderPath(shader_label);

    fragment_mutex_.lock();
    if (shaders_.find(shader_path) == shaders_.end()) {
        shaders_[shader_path] = std::make_unique<FragmentGlShader>(shader_path, program_id);
    }
    fragment_mutex_.unlock();

    assert(shaders_.find(shader_path) != shaders_.end());

    return shaders_[shader_path].get();
}

GlShaderFactory::~GlShaderFactory() noexcept
{
    sInstanciated = false;
}

}  // namespace renderer

}  // namespace nextfloor