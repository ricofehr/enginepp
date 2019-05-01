/**
 *  @file global_timer.h
 *  @brief Timer Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GLOBALTIMER_H_
#define NEXTFLOOR_CORE_GLOBALTIMER_H_

// #include <GL/glew.h>
// #include <GLFW/glfw3.h>

#include <memory>
#include <chrono>

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
 *  @class GlobalTimer
 *  @brief GlobalTimer maintains global timer management for the program.\n
 *  Implement Singleton Pattern which ensure a sole Timer object for the program
 */
class GlobalTimer {

public:

    /**
     *  Default Move constructor
     */
    GlobalTimer(GlobalTimer&&) = default;

    /**
     *  Default Move assignment
     */
    GlobalTimer& operator=(GlobalTimer&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    GlobalTimer(const GlobalTimer&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    GlobalTimer& operator=(const GlobalTimer&) = delete;

    /**
     *  Return sole Instance
     *  @return sole ConfigEngine instance
     */
    static GlobalTimer* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new GlobalTimer;
        return instance;
    }

    /**
     *  Time Iteration
     */
    static void LoopTimer()
    {
        Instance()->Loop();
    }

    /** Time elapsed since last iteration */
    static double sDeltaTime;

    /** Time elapsed since program beginning */
    static double sTotalTime;

protected:

    /**
     *  Default Constructor
     *  With a protected scope for avoid any directly Allocation
     */
    GlobalTimer() = default;

private:

    /**
     *  Compute Time
     */
    void Loop()
    {
        using namespace std::chrono;
        static double sLastTime = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();

        /* Delta time */
        double current_time = duration_cast<duration<double>>(high_resolution_clock::now().time_since_epoch()).count();
        sDeltaTime = current_time - sLastTime;
        sLastTime = current_time;
        sTotalTime += sDeltaTime;
    }
};


} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GLOBALTIMER_H_
