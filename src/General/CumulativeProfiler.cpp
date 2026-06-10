#include "CumulativeProfiler.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Entry {
    double totalSeconds = 0.0;
    uint64_t count = 0;
};

std::vector<std::string> g_Names;
std::unordered_map<std::string, int> g_NameToId;
std::vector<Entry> g_Entries;

int internName(const char* name)
{
    std::string key(name);
    auto it = g_NameToId.find(key);
    if (it != g_NameToId.end())
        return it->second;
    int id = (int)g_Names.size();
    g_Names.push_back(key);
    g_NameToId[key] = id;
    g_Entries.push_back({});
    return id;
}

} // namespace

int CumulativeProfiler::intern(const char* name)
{
    return internName(name);
}

void CumulativeProfiler::reset()
{
    g_Names.clear();
    g_NameToId.clear();
    g_Entries.clear();
}

void CumulativeProfiler::addSample(int id, double seconds)
{
    if (id < 0 || id >= (int)g_Entries.size())
        return;
    Entry& e = g_Entries[id];
    e.totalSeconds += seconds;
    e.count++;
}

void CumulativeProfiler::addSample(const char* name, double seconds)
{
    addSample(internName(name), seconds);
}

void CumulativeProfiler::printReport(const char* title)
{
    if (g_Entries.empty())
        return;

    std::vector<int> order;
    order.reserve(g_Entries.size());
    for (int i = 0; i < (int)g_Entries.size(); i++)
    {
        if (g_Entries[i].count > 0)
            order.push_back(i);
    }

    std::sort(order.begin(), order.end(), [](int a, int b) {
        return g_Entries[a].totalSeconds > g_Entries[b].totalSeconds;
    });

    double grandTotal = 0.0;
    for (int id : order)
        grandTotal += g_Entries[id].totalSeconds;

    std::cout << "\n=== Cumulative profile";
    if (title && title[0])
        std::cout << ": " << title;
    std::cout << " ===\n";

    for (int id : order)
    {
        const Entry& e = g_Entries[id];
        double ms = e.totalSeconds * 1000.0;
        double pct = grandTotal > 0.0 ? (e.totalSeconds / grandTotal) * 100.0 : 0.0;
        double avgUs = e.count > 0 ? (e.totalSeconds / (double)e.count) * 1e6 : 0.0;
        std::cout << "  " << g_Names[id]
                  << ": " << ms << " ms (" << pct << "%)"
                  << "  calls=" << e.count
                  << "  avg=" << avgUs << " us\n";
    }
    std::cout << "  (sum of buckets: " << grandTotal * 1000.0
              << " ms; nested buckets overlap)\n\n";
}

CumulativeProfiler::Scope::Scope(const char* name)
    : m_Id(internName(name))
{
    m_Sw.start();
}

CumulativeProfiler::Scope::Scope(int id)
    : m_Id(id)
{
    m_Sw.start();
}

CumulativeProfiler::Scope::~Scope()
{
    addSample(m_Id, m_Sw.stop_time());
}
