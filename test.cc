#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <vector>
#include <algorithm>

// Struct to store CPU core statistics
struct CpuCore {
    std::string name;
    long user;
    long nice;
    long system;
    long idle;
    long iowait;
    long irq;
    long softirq;
    long steal;
    long guest;
    long guest_nice;
};

// Function to parse CPU statistics from /proc/stat
std::vector<CpuCore> parseCpuStats(const std::string& filename) {
    std::vector<CpuCore> cores;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cores;
    }

    std::string line;
    // Updated regex: match cpu or cpuX followed by multiple spaces or tabs
    std::regex cpu_regex(R"(cpu(\d*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+))");

    while (std::getline(file, line)) {
        // Remove any trailing \r (carriage return)
        if (!line.empty() && line[line.size() - 1] == '\r') {
            line = line.substr(0, line.size() - 1);
        }

        // Debug: Show the exact line read
        std::cout << "Read line: \"" << line << "\"" << std::endl;

        // Remove extra spaces (collapse multiple spaces into one)
        std::string clean_line;
        std::unique_copy(line.begin(), line.end(), std::back_inserter(clean_line), 
                          [](char lhs, char rhs) { return std::isspace(lhs) && std::isspace(rhs); });
        // Trim leading and trailing spaces
        clean_line.erase(0, clean_line.find_first_not_of(" \t"));
        clean_line.erase(clean_line.find_last_not_of(" \t") + 1);

        // Match the cleaned line with the regex
        std::smatch match;
        if (std::regex_match(clean_line, match, cpu_regex)) {
            CpuCore core;
            core.name = match[1].str().empty() ? "cpu" : "cpu" + match[1].str(); // "cpu" for total stats, cpu0, cpu1, etc.
            core.user = std::stol(match[2].str());
            core.nice = std::stol(match[3].str());
            core.system = std::stol(match[4].str());
            core.idle = std::stol(match[5].str());
            core.iowait = std::stol(match[6].str());
            core.irq = std::stol(match[7].str());
            core.softirq = std::stol(match[8].str());
            core.steal = std::stol(match[9].str());
            core.guest = std::stol(match[10].str());
            core.guest_nice = 0; // Guest nice is typically 0 in most systems.

            cores.push_back(core);
        } else {
            std::cout << "No match for line: \"" << clean_line << "\"" << std::endl;
        }
    }
    return cores;
}

// Function to print the CPU stats
void printCpuStats(const std::vector<CpuCore>& cores) {
    for (const auto& core : cores) {
        std::cout << core.name << ": "
                  << "User=" << core.user << " "
                  << "Nice=" << core.nice << " "
                  << "System=" << core.system << " "
                  << "Idle=" << core.idle << " "
                  << "IOWait=" << core.iowait << " "
                  << "IRQ=" << core.irq << " "
                  << "SoftIRQ=" << core.softirq << " "
                  << "Steal=" << core.steal << " "
                  << "Guest=" << core.guest << std::endl;
    }
}

int main() {
    std::vector<CpuCore> cores = parseCpuStats("/proc/stat");
    if (cores.empty()) {
        std::cerr << "No CPU stats found or failed to parse /proc/stat." << std::endl;
    } else {
        printCpuStats(cores);
    }
    return 0;
}
