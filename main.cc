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


    cout << "Spawning Terminal UI Thread... ";
	std::thread terminalUIThread = std::thread( [&] () {
	 // this is the master thread
        int iterations = 0;
		auto tStart = high_resolution_clock::now();
			
        // while ( true ) {
            auto document = std::function( [&] () {
                Elements temp, accum;
                for ( int j = 0; j < 6; j++ ) {
                    for ( int i = 0; i < 12; i++ ) {
                        int idx = i + 6 * j;
                        float intensity = usagePercentage[ idx ];
                        temp.push_back( text( " " ) | bgcolor( Color::RGB( 255 * sqrt( intensity ), 128 * intensity, 0 ) ) );
                    }
                    accum.push_back( hbox( temp ) );
                    temp.resize( 0 );
                }
                return vbox( accum );
            } ) () | borderHeavy;

            // wip need to figure out hwo to create an unwindowed interactive thing
            auto screen = Screen::Create( Dimension::Fit( document ), Dimension::Fit( document ) );
            Render( screen, document );
            screen.Print();
            cout << endl;

            Crystal c;

            // do the terminal UI shit
                // CPU activity
                // display table of Crystal activity

            sleep_for( 20s );
        // }
            
        // cout << "Running Monitor Thread " << std::chrono::duration_cast< std::chrono::milliseconds >( high_resolution_clock::now() - tStart )  << "ms" << endl;
        threadKill = true;
    });            
    cout << "Done." << endl;

    procUpdaterThread.join();
    terminalUIThread.join();
}