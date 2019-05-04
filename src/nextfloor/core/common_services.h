/**
 *  @file common_services.h
 *  @brief CommonServices Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_COMMONSERVICES_H_
#define NEXTFLOOR_CORE_COMMONSERVICES_H_

#include "nextfloor/core/global_config.h"
#include "nextfloor/core/file_io.h"
#include "nextfloor/core/global_timer.h"
#include "nextfloor/core/random_number.h"
#include "nextfloor/core/global_log.h"
#include "nextfloor/core/exit_program.h"

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::core
 *  @brief Core system
 */
namespace core {

/**
 *  @class CommonServices
 *  @brief CommonServices is an unique entry point qho provides global services to other program modules\n
 *  Implement ServiceLocator Pattern
 */
class CommonServices {

public:
    static GlobalConfig& getConfig() { return config_; }
    static const FileIO& getFileIO() { return file_io_; }
    static GlobalTimer& getTimer() { return timer_; }
    static const GlobalLog& getLog() { return log_; }
    static const RandomNumber& getRandomNumber() { return random_number_; }
    static const ExitProgram& getExit() { return exit_; }

private:
    static GlobalConfig config_;
    static FileIO file_io_;
    static GlobalTimer timer_;
    static GlobalLog log_;
    static RandomNumber random_number_;
    static ExitProgram exit_;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_COMMONSERVICES_H_
