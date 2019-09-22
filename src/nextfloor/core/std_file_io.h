/**
 *  @file std_file_io.h
 *  @brief File I/O Operations
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_STDFILEIO_H_
#define NEXTFLOOR_CORE_STDFILEIO_H_

#include "nextfloor/core/file_io.h"

#include <fstream>
#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class GenericFileIO
 *  @brief contains standard FileIO Operations for the engine
 */
class StdFileIO : public FileIO {

public:
    StdFileIO();
    ~StdFileIO() noexcept final;

    std::string ReadFile(const std::string& file_path) const final;

private:
    void HandleErrorOnReadFile(const std::string& file_path) const;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_STDFILEIO_H_
