/**
 *  @file gl_renderer.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLRENDERER_H_
#define NEXTFLOOR_RENDERER_GLRENDERER_H_

#include "nextfloor/gameplay/renderer_engine.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

#include "nextfloor/renderer/pipeline_program.h"


namespace nextfloor {

namespace renderer {

class GlRendererEngine : public nextfloor::gameplay::RendererEngine {

public:
    ~GlRendererEngine() override = default;

protected:
    GlRendererEngine(PipelineProgram* pipeline_program);

    GlRendererEngine(GlRendererEngine&&) = default;
    GlRendererEngine& operator=(GlRendererEngine&&) = default;
    GlRendererEngine(const GlRendererEngine&) = delete;
    GlRendererEngine& operator=(const GlRendererEngine&) = delete;

    PipelineProgram* pipeline_program_{nullptr};
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLRENDERER_H_