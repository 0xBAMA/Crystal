#include "crystal.h"
#include "reporter.h"

// master kill for all threads
atomic< bool > threadKill = false;

int main ( int argc, char** argv ) {
    // Proc Updater Thread
        // make sure the proc header has "previous" data
    updateProcData();
    sleep_for( 5ms );
    updateProcData();

    // functionally a "service" thread, to keep the proc data updated
    cout << "Spawning Proc Updater Thread... ";
    std::thread procUpdaterThread = std::thread( [&]() {
        while ( !threadKill ) {    
            updateProcData();
            sleep_for( 5ms );
        }
    });
    cout << "Done." << endl;

    // tbd...
    Crystal c;

    cout << "Spawning Terminal UI Thread... ";
	std::thread terminalUIThread = std::thread( [&] () {
	 // this is the master thread
        int iterations = 0;
		auto tStart = high_resolution_clock::now();
			
        // while ( true ) {

            // do the terminal UI shit
                // CPU activity
                // display table of Crystal activitybhgggghj
        
            // sleep_for( 100ms );            
            // cout << "Running Monitor Thread " << std::chrono::duration_cast< std::chrono::milliseconds >( high_resolution_clock::now() - tStart )  << "ms" << endl;
        // }
    });            
    cout << "Done." << endl;

    procUpdaterThread.join();
    terminalUIThread.join();
}