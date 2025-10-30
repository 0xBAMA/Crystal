#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unistd.h> // For sleep()

// Structure to store CPU data
struct CpuData {
    std::string name;
    int index;
    long idleTicks;
    long busyTicks;
};

std::vector< CpuData > lCpus;  // Store last readings
std::vector< CpuData > cCpus;  // Store current readings
std::vector< float > usagePercentage;

void updateProcData () {
    static size_t lCount = 0; // Loop counter
    std::ifstream statFile( "/proc/stat" );
    std::string line;

    cCpus.resize( 0 );
    cCpus.reserve( 256 );

    while ( std::getline( statFile, line ) ) {
        std::istringstream ss( line );
        std::vector< std::string > tokens;
        std::string token;

        // Tokenize the line into strings by spaces
        while ( ss >> token ) tokens.push_back( token );

        // Check if this is a CPU line ( starts with "cpu" but not "cpu" itself )
        if ( tokens[ 0 ].substr( 0, 3 ) == "cpu" && tokens[ 0 ] != "cpu" ) {
            CpuData cCpu;
            cCpu.name = tokens[ 0 ];  // CPU name (cpu0, cpu1, ...)
            cCpu.index = cCpus.size(); // CPU index
            cCpu.idleTicks = 0;
            cCpu.busyTicks = 0;

            // Parse CPU idle (4th column) and busy (3rd, 5th, etc.) ticks
            for ( size_t i = 1; i < tokens.size(); ++i ) {
                long value = std::stol( tokens[ i ] );
                if ( i == 3 || i == 4 ) { // Idle and I/O wait
                    cCpu.idleTicks += value;
                } else { // Other values are considered busy
                    cCpu.busyTicks += value;
                }
            }

            cCpus.push_back( cCpu );  // Store current CPU data
        }
    }

    // Ensure lCpus and cCpus have the same size
    if ( lCount == 0 ) {
        lCpus.resize( cCpus.size() );
        usagePercentage.resize( cCpus.size() );
    }

    // Include the last CPU reading
    for ( size_t i = 0; i < cCpus.size(); ++i ) {
        CpuData& cCpu = cCpus[ i ];
        if ( lCount > 0 ) {
            // Compare the current and last CPU readings
            CpuData& lCpu = lCpus[ i ];
            float dTotal = ( cCpu.idleTicks + cCpu.busyTicks ) - ( lCpu.idleTicks + lCpu.busyTicks );
            float dUsed = dTotal - ( cCpu.idleTicks - lCpu.idleTicks );
            if ( dTotal == 0 ) dTotal = 1;  // Avoid division by 0

            // store CPU usage
            usagePercentage[ i ] = 0.6f * usagePercentage[ i ] + 0.4f * ( dUsed / dTotal );
            // cout << "usage: " << i << " " << usagePercentage[ i ] << endl;
        }

        // Store the current reading for next iteration
        lCpus[ i ] = cCpu;
    }

    // ready for the next interval
    ++lCount;
}