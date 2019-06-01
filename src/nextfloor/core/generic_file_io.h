/**
 *  @file file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GENERICFILEIO_H_
#define NEXTFLOOR_CORE_GENERICFILEIO_H_

#include "nextfloor/core/file_io.h"

#include <fstream>
#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class GenericFileIO
 *  @brief contains standard FileIO Operations for the engine
 */
class GenericFileIO : public FileIO {

public:

    GenericFileIO();

    GenericFileIO(GenericFileIO&&) = default;

    GenericFileIO& operator=(GenericFileIO&&) = default;

    /**
     *  Copy constructor Deleted : Ensure a sole Instance
     */
    GenericFileIO(const GenericFileIO&) = delete;

    /**
     *  Copy assignement Deleted: Ensure a sole Instance
     */
    GenericFileIO& operator=(const GenericFileIO&) = delete;

    virtual ~GenericFileIO() override final;

    virtual std::string ReadFile(std::string file_path) const override final;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GENERICFILEIO_H_
