#include <cmath>

#include "crystal.h"
#include "reporter.h"

// master kill for all threads
atomic< bool > threadKill = false;

// support stuff for the ui
auto screen = ScreenInteractive::FixedSize( 100, 36 );

// list of crystals
constexpr int numCrystalsMax = 8;
int numCrystals = 0;
shared_ptr< Crystal > crystals[ numCrystalsMax ];

int left_size = 36;
std::chrono::time_point< std::chrono::system_clock > tStart;
Component GetUpdatedMenuComponent () {
    return CatchEvent( ResizableSplit({
        .main = Container::Vertical({
             Container::Horizontal({ Renderer( [&] () {
                 Elements temp, accum;
                 for ( int j = 0; j < 6; j++ ) {
                     for ( int i = 0; i < 12; i++ ) {
                         int idx = i + 6 * j;
                         float intensity = usagePercentage[ idx ];
                         temp.push_back( text( " " ) | bgcolor( Color::RGB( 255 * std::sqrt( intensity ), 128 * intensity, 0 ) ) );
                     }
                     accum.push_back( hbox( temp ) );
                     temp.resize( 0 );
                 }
                 return vbox( accum );
             }) | borderHeavy | flex_shrink,
                 Container::Vertical({
                    Button({
                        .label = " Add Crystal ",
                        .on_click = [&] () {
                            for ( int i = 0; i < numCrystalsMax; i++ ) {
                                if ( crystals[ i ] == nullptr ) { // take the first open slot
                                    crystals[ i ] = make_shared< Crystal >();
                                    break;
                                }
                                // if you don't find a null pointer, we are full, do not allocate a new crystal
                            }
                        },
                        .transform = ButtonOption::Ascii().transform,
                    }),
                    Button({
                        .label = " Quit ",
                        .on_click = screen.ExitLoopClosure(),
                        .transform = ButtonOption::Ascii().transform,
                    }),
                 } ) | borderRounded | flex_grow } ),
             // Slider( "Temperature", &temperature, 0.0f, 69.0f ),
             Renderer( [&] ( bool focused ) {
                 auto c1 = color( Color::RGB( 255, 34, 0 ) );
                 auto c2 = color( Color::RGB( 255, 255, 34 ) );
                 return vbox({
                     hbox({ text( "Uptime:              " ) | c1, text( std::to_string( float( std::chrono::duration< float, std::milli >( high_resolution_clock::now() - tStart ).count() ) / 1000.0f ) ) | c2, text( "s" ) | c1 }),
                     // hbox({ text( "Job Counter:         " ) | c1, text( std::to_string( jobCounter.load() ) ) | c2 }),
                     // hbox({ text( "Anchored Particles:  " ) | c1, text( std::to_string( numAnchored ) ) | c2 }),
                     hbox({ text( "Extents:" ) | c1 }),
                     // hbox({ text( " x:  " ) | c1, text( to_string( minExtents.x ) + " " ) | c2, text( to_string( maxExtents.x ) ) | c2 }),
                     // hbox({ text( " y:  " ) | c1, text( to_string( minExtents.y ) + " " ) | c2, text( to_string( maxExtents.y ) ) | c2 }),
                     // hbox({ text( " z:  " ) | c1, text( to_string( minExtents.z ) + " " ) | c2, text( to_string( maxExtents.z ) ) | c2 }),
                     // hbox({ text( "Frames Queued: " ) | c1, text( to_string( pointerPoolAllocator / particlesPerStep ) ) | c2 }),
                 });
             })
        }),
        .back = [&] () {
            Components c;
            c.push_back( Container::Vertical( { Renderer( [] () { return text( "  " ); } ) } ) );
            for ( int i = 0; i < numCrystalsMax; i++ ) {
                const int iC = i;
                c.push_back( Container::Vertical({
                    Renderer([&, iC] {
                        if ( crystals[ iC ] != nullptr ) {
                            // this actually needs to prepare a status report with the crystal state
                            const float percentageComplete = crystals[ iC ]->GetPercentage();
                            const string stateString = crystals[ iC ]->GetStateString();
                            return hbox({ text( to_string( iC ) + ": " + stateString + " " ), gauge( percentageComplete ), });
                        }
                        return vbox({
                            hbox({ text( "  " ) }),
                            hbox({ text( "Crystal not found." ) | color( Color::RGB( 64, 64, 64 ) ) }),
                            hbox({ text( "  " ) }),
                        });
                    }),
                    Container::Horizontal({
                        Button( " Save ", [ &, iC ] () { crystals[ iC ]->Save(); }, ButtonOption::Ascii() ) | Maybe( [ &, iC ]{ return crystals[ iC ] != nullptr; }),
                        Renderer( []() { return text( "      " ); } ),
                        Button( " Screenshot ", [ &, iC ] () { crystals[ iC ]->Screenshot(); }, ButtonOption::Ascii() ) | Maybe( [ &, iC ]{ return crystals[ iC ] != nullptr; }),
                    }) | align_right,
                }));
            }
            return Container::Vertical( c );
        }(),
        .direction = Direction::Left,
        .main_size = &left_size,
        .separator_func = [] { return separatorHeavy(); },
    }) | borderHeavy,
[&]( Event event ) {
        if ( event == Event::Character( 'q' ) ) {
            screen.ExitLoopClosure()();
            return true;
        }

        // if ( event == Event::Character('w') ) {
        //     // prepareOutputScreenshot();
        //     // prepareOutputScreenshotDeltaTracking();
        //     return true;
        // }

        return false;
    });
}

Component componentHandle;

int main ( int argc, char** argv ) {
	tStart = high_resolution_clock::now();

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
	    componentHandle = GetUpdatedMenuComponent();

        cout << "Entering Main Loop..." << endl;
        ftxui::Loop loop( &screen, componentHandle );
        bool quit = false;
        while ( !loop.HasQuitted() && !quit ) {
            screen.RequestAnimationFrame();
            loop.RunOnce();
            sleep_for( 100ms );
        }

        // cout << "Running Monitor Thread " << std::chrono::duration_cast< std::chrono::milliseconds >( high_resolution_clock::now() - tStart )  << "ms" << endl;
        threadKill = true;
    });

    procUpdaterThread.join();
    terminalUIThread.join();
}
