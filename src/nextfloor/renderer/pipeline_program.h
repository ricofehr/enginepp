/**
 *  @file shader.h
 *  @brief PipelineProgram abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_PIPELINEPROGRAM_H_
#define NEXTFLOOR_RENDERER_PIPELINEPROGRAM_H_


namespace nextfloor {

namespace renderer {

/**
 *  PipelineProgram Interface class, define design for all program shaders
 */
class PipelineProgram {

public:
    virtual ~PipelineProgram() = default;

    virtual unsigned int getMatrixId() const = 0;
    virtual unsigned int getProgramId() const = 0;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_PIPELINEPROGRAM_H_