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

    StdFileIO(StdFileIO&&) = default;
    StdFileIO& operator=(StdFileIO&&) = default;
    StdFileIO(const StdFileIO&) = delete;
    StdFileIO& operator=(const StdFileIO&) = delete;

    virtual ~StdFileIO() override final;

    virtual std::string ReadFile(std::string file_path) const override final;

private:
    void HandleErrorOnReadFile(std::string file_path) const;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_STDFILEIO_H_
