/**
 *  @file shader.h
 *  @brief Shader abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SHADER_H_
#define NEXTFLOOR_RENDERER_SHADER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace renderer
 *  @brief rederer namespace
 */
namespace renderer {

/**
 *  Shader Interface class, define design for all shader loaders
 */
class Shader {

public:

    /**
     *  Default Move constructor
     */
    Shader(Shader&&) = default;

    /**
     *  Default Move assignment
     */
    Shader& operator=(Shader&&) = default;

    /**
     *  Default Copy constructor
     */
    Shader(const Shader&) = default;

    /**
     *  Default Copy assignment
     */
    Shader& operator=(const Shader&) = default;

    /**
     *  Abstract Class, define virtual destructor
     */
    virtual ~Shader() = default;

    /**
     *  Load the Shader
     */
    virtual void LoadShader() = 0;

    /**
     *  Link the Shader
     */
    virtual void LinkShader();

    /**
     *  Link the Shader
     */
    virtual void DetachShader();

    /**
     *  Check Program
     */
    static void CheckProgram();

    /**
     *  Accessor
     */
    GLuint shader_id() { return shader_id_; }

    /** Global GL Program ID */
    static GLuint sProgramId;

protected:

    /**
     *  Default Construtor
     *  Protected scope ensures Abstract Class Design
     */
    Shader(std::string shader_filepath);

    /** Shader relative path */
    std::string shader_filepath_;

    /** Shader GL ID */
    GLuint shader_id_;

};

} // namespace renderer

} // namespace nextfloor

#endif // NEXTFLOOR_RENDERER_SHADER_H_