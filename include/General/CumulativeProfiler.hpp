#ifndef CUMULATIVE_PROFILER_HPP
#define CUMULATIVE_PROFILER_HPP

#include "Stopwatch.hpp"
#include <cstdint>

/** Accumulates named timing samples; print with printReport(). */
class CumulativeProfiler {
public:
    static int intern(const char* name);
    static void reset();
    static void addSample(int id, double seconds);
    static void addSample(const char* name, double seconds);
    static void printReport(const char* title);

    class Scope {
    public:
        explicit Scope(const char* name);
        explicit Scope(int id);
        ~Scope();
    private:
        int m_Id;
        Stopwatch m_Sw;
    };
};

#define PROFILE_SCOPE(name) CumulativeProfiler::Scope _profile_scope_##__LINE__(name)

#endif
