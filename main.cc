#include <cmath>

#include "crystal.h"
#include "reporter.h"

// master kill for all threads
atomic< bool > threadKill = false;

// support stuff for the ui
auto screen = ScreenInteractive::FixedSize( 80, 20 );
std::string saveString = "Save";
std::string ssString = "Screenshot";

// list of crystals
constexpr int numCrystalsMax = 8;
bool showCrystal[ numCrystalsMax ] = { false };
vector< shared_ptr< Crystal > > crystals;

// struct crystalListItem {
//     // we need to maintain the button component and the renderer component
//     Component buttons;
//     Component renderer;
// };
//
// vector< crystalListItem > crystalListItems;
//
// crystalListItem makeCrystalListItem() {
//     crystalListItem c;
//     const int idx = crystals.size() - 1;
//     shared_ptr< Crystal > crystal = crystals[ idx ];
//     c.buttons = Container::Horizontal({
//         Button( &saveString, [&](){ crystal->Save(); } ),
//         Button( &ssString, [&](){ crystal->Screenshot(); } )
//     });
//     c.renderer = Container::Vertical({ Renderer( c.buttons, [&] () {
//         return vbox({
//             hbox({ text( "Crystal " + to_string( idx ) ) | center, border( gauge( crystals[ idx ]->percentageCompletion ) ) }),
//             c.buttons->Render()
//             });
//         })
//     });
//     return c;
// }

// Component GetCrystalList() {
    // Components c;
    //
    // for ( int i = 0; i < crystals.size(); i++ ) {
    //     auto buttons =
    //
    //  c.push_back( Container::Vertical( {
    //     Renderer([&] {
    //         return( vbox({
    //             hbox({ text( "Crystal " + to_string( i ) ) | center, border( gauge( crystals[ i ]->percentageCompletion ) ) }),
    //             hbox(
    //             );
    // } ),
    //      Container::Horizontal({
    //          Button({
    //              .label = "Button 1",
    //              .on_click = [&](){},
    //          }),
    //          Button({
    //              .label = "Button 2",
    //              .on_click = [&](){},
    //          }),
    //      })
    //  } ) );
    // }
    // return Container::Vertical( c );


// auto buttons = Container::Horizontal({ Button(&label, screen.ExitLoopClosure()), Button(&label2, screen.ExitLoopClosure()) });
// auto renderer = Renderer( buttons, [&] {
  // return hbox({
    // text("A button:"),
    // buttons->Render(),
  // });
// });
// screen.Loop(renderer);
// }

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

        // Crystal c;

	    // some test config flags
	    bool checked[3] = { false, false, false };
        cout << "Done." << endl;
        // float temperature = 0.0f;

	    // list of status + buttons to control
        int left_size = 36;
        auto component = ResizableSplit({
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
                        // Checkbox( "Check me", &checked[ 0 ] ),
                        // Checkbox( "Check me", &checked[ 1 ] ),
                        // Checkbox( "Check me", &checked[ 2 ] ),
                        Button({
                            .label = "Add Crystal",
                            .on_click = [&] () {
                                // this also needs to update the menu stuff with the information for the new crystal
                                crystals.push_back( make_shared< Crystal >() );
                                // crystals.
                            },
                        }),
                        Button({
                           .label = "Quit",
                           .on_click = screen.ExitLoopClosure(),
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
                for ( int i = 0; i < numCrystalsMax; i++ ) {
                    const int iC = i;
                    c.push_back( Container::Vertical({
                        Renderer([&] {
                            return( vbox({
                                hbox({ text( "Crystal " + to_string( iC ) ) | center, border( gauge( crystals[ iC ]->percentageCompletion ) ) } ),
                            } ) );
                        }),
                        Container::Horizontal({
                            Button( &saveString, [&] () { crystals[ iC ]->Save(); } ),
                            Button( &ssString, [&] () { crystals[ iC ]->Screenshot(); } )
                        })
                    }) | Maybe( [&] () { return iC < crystals.size(); } ) );
                }
                return Container::Vertical(c );
            }(),
            .direction = Direction::Left,
            .main_size = &left_size,
            .separator_func = [] { return separatorHeavy(); },
        });

         // do the terminal UI shit
         // CPU activity
         // display table of Crystal activity

        auto ftxDAG = CatchEvent( component, [&]( Event event ) {
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

        cout << "Entering Main Loop..." << endl;
        ftxui::Loop loop( &screen, ftxDAG );
        bool quit = false;
        while ( !loop.HasQuitted() && !quit ) {
             screen.RequestAnimationFrame();
             loop.RunOnce();
             sleep_for( 100ms );

             // if ( numAnchored >= maxParticles ) {
             // quit = true;
             // }
         }

         // sleep_for( 20s );
         // }

         // cout << "Running Monitor Thread " << std::chrono::duration_cast< std::chrono::milliseconds >( high_resolution_clock::now() - tStart )  << "ms" << endl;
         threadKill = true;
    });

    procUpdaterThread.join();
    terminalUIThread.join();
}
