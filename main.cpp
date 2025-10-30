//=================================================================================================
// command line output
#include <string>
#include <iostream>
using std::cout;
using std::endl;
using std::string;
//=================================================================================================
// basic containers
#include <vector>
using std::vector;
//=================================================================================================
#include <fstream>
//=================================================================================================
#include <unordered_map>
using std::unordered_map;
//=================================================================================================
// timekeeping
#include <chrono>
using namespace std::chrono_literals;
//=================================================================================================
// mutex / lock stuff
#include <mutex>
using std::mutex;
using std::lock_guard;
//=================================================================================================
// thread stuff
#include <thread>
using std::thread;
using std::this_thread::sleep_for;
//=================================================================================================
// shared_ptr
#include <memory>
using std::make_unique;
using std::unique_ptr;
//=================================================================================================
// specific width types + atomics, for "program counter"
#include <cstdint>
#include <atomic>
using std::uintmax_t;
using std::atomic_uintmax_t;
//=================================================================================================
// Terminal UI output
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>

#include <ftxui/component/loop.hpp>
#include <ftxui/component/captured_mouse.hpp>  // for ftxui
#include <ftxui/component/component.hpp>  // for Checkbox, Renderer, Vertical
#include <ftxui/component/component_base.hpp>      // for ComponentBase
#include <ftxui/component/screen_interactive.hpp>  // for ScreenInteractive
using namespace ftxui;
//=================================================================================================
// math/vector stuff
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp> 					// glm::vec3
#include <glm/vec4.hpp> 					// glm::vec4
#include <glm/mat4x4.hpp> 					// glm::mat4
#include <glm/ext/matrix_transform.hpp>		// glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp>	// glm::perspective
#include <glm/ext/scalar_constants.hpp>		// glm::pi
#include <glm/gtx/string_cast.hpp>
using glm::vec3;
using glm::vec4;
using glm::mat4;
using glm::pi;
using glm::ivec3;
using glm::dot;

// key hash needed for std::unordered_map with ivec3 keys
namespace std {
	template<> struct hash< ivec3 > {
		// custom specialization of std::hash can be injected in namespace std
		std::size_t operator()( ivec3 const& s ) const noexcept {
			std::size_t h1 = std::hash< int >{}( s.x );
			std::size_t h2 = std::hash< int >{}( s.y );
			std::size_t h3 = std::hash< int >{}( s.z );
			return h1 ^ ( h2 << 4 ) ^ ( h3 << 8 );
		}
	};
}
//=================================================================================================
// random number generation utilities
#include <random>

// Float version ( uniform distribution )
class rng {
public:

	// hardware seeded
	rng ( float lo, float hi ) :
		distribution( std::uniform_real_distribution< float >( lo, hi ) ) {
			std::random_device r;
			std::seed_seq seed { r(),r(),r(),r(),r(),r(),r(),r(),r() };
			generator = std::mt19937_64( seed );
		}

	// known 32-bit seed value
	rng ( float lo, float hi, uint32_t seed ) :
		generator( std::mt19937_64( seed ) ),
		distribution( std::uniform_real_distribution< float >( lo, hi ) ) {}

	// get the value
	// float get () { return distribution( generator ); }
	float operator () () { return distribution( generator ); }

private:
	std::mt19937_64 generator;
	std::uniform_real_distribution< float > distribution;
};

// Float version ( normal distribution )
class rngN {
public:

	// hardware seeded
	rngN ( float center, float width ) :
		distribution( std::normal_distribution< float >( center, width ) ) {
			std::random_device r;
			std::seed_seq seed { r(),r(),r(),r(),r(),r(),r(),r(),r() };
			generator = std::mt19937_64( seed );
		}

	// known 32-bit seed value
	rngN ( float center, float width, uint32_t seed ) :
		generator( std::mt19937_64( seed ) ),
		distribution( std::normal_distribution< float >( center, width ) ) {}

	// get the value
	float operator () () { return distribution( generator ); }

private:
	std::mt19937_64 generator;
	std::normal_distribution< float > distribution;
};

// Integer version
class rngi {
public:

	// hardware seeded
	rngi ( int lo, int hi ) :
		distribution( std::uniform_int_distribution< int >( lo, hi ) ) {
			std::random_device r;
			std::seed_seq seed { r(),r(),r(),r(),r(),r(),r(),r(),r() };
			generator = std::mt19937_64( seed );
		}

	// known 32-bit seed value
	rngi ( int lo, int hi, uint32_t seed ) :
		generator( std::mt19937_64( seed ) ),
		distribution( std::uniform_int_distribution< int >( lo, hi ) ) {}

	// get the value
	// int get () { return distribution( generator ); }
	int operator () () { return distribution( generator ); }

private:
	std::mt19937_64 generator;
	std::uniform_int_distribution< int > distribution;
};
//=================================================================================================
//=================================================================================================

// going to use this as a threshold for a kind of "bonding affinity"... may need more detail here than 1 degree of freedom
// float transformSimilarity ( const particle_t &a, const particle_t &b ) const {
    // applying the transform, taking advantage of homogeneous coordinates to only get the orientation of the transform
    // vec3 base = vec3( 1.0f, 0.0f, 0.0f );
    // vec3 tA = ( a.transform * vec4( base, 0.0f ) ).xyz();
    // vec3 tB = ( b.transform * vec4( base, 0.0f ) ).xyz();
    // this should probably get a remapping that peaks at 0 and falls off sharply, like the cauchy thing
    // return tan( dot( tA, tB ) * pi * 2.0f );
// }

struct gridCell {
    // keeping "count" as an atomic, separate from .size()...
        // this means we can make sure that the mat4 is fully constructed before showing as available to other threads
    std::atomic< std::int32_t > count;
    std::vector< unique_ptr< const mat4 > > particles;
};

// the list of particles which have been anchored
unordered_map< ivec3, gridCell > anchoredParticles;
mutex anchoredParticlesGuard;

// maintaining stats on the above container
ivec3 minExtents = ivec3( 0 );
ivec3 maxExtents = ivec3( 0 );
int numAnchored = 0;

// the particles are the diffusion-limit mechanism
vector< vec4 > particlePool;

// get a point on the boundary
void respawnParticle ( vec4 p ) {
    rngi pick( 0, 2 );
    rng pick2( 0.0f, 1.0f );

    bool face = ( pick2() < 0.5f );
    constexpr float margin = 10.0f;
    p.x = glm::mix( minExtents.x - margin, maxExtents.z + margin, pick2() );
    p.y = glm::mix( minExtents.y - margin, maxExtents.y + margin, pick2() );
    p.z = glm::mix( minExtents.z - margin, maxExtents.z + margin, pick2() );

    switch ( pick() ) {
        // we will be on one of the parallel x faces, flatten
        case 0: p.x = face ? minExtents.x - margin : maxExtents.x + margin; break;

        // ditto, y faces
        case 1: p.y = face ? minExtents.y - margin : maxExtents.y + margin; break;

        // z faces
        case 2: p.z = face ? minExtents.z - margin : maxExtents.z + margin; break;

        // shouldn't hit, but will be uniform random spawn if you do
        default:
            break;
    }

    // the .w will be a counter... it is decremented any update during which it is outide the current bounding volume
    p.w = 100.0f; // I think this is a nice mechanism - it can wander, but in a bounded way - when it hits zero, respawn
}

void anchorParticle ( vec3 p, const mat4 pTransform ) {
    // we need to anchor this particle... but first, lock the mutex so only one thread can do this at once
    lock_guard< mutex > lock( anchoredParticlesGuard ); // mutex object entering scope locks... declaration is blocking 

    // precompute integer rounded coordinates
    const ivec3 iP = ivec3( p );

    // push the constructed mat4 onto the list
    anchoredParticles[ iP ].particles.push_back( make_unique< const mat4 >( pTransform ) );

    // now that it's safe to access, show it as available for other threads
    anchoredParticles[ iP ].count++; // "++" operator does the atomic increment

    // tracking how many particles have been added, and the min and max extents
    minExtents = glm::min( iP, minExtents );
    maxExtents = glm::max( iP, maxExtents );
    numAnchored++;

    // and the mutex leaves scope, unblocking for the next write access
}

// this is the update, operating on a particular particle
void particleUpdate ( uintmax_t jobIndex ) {
    uintmax_t idx = jobIndex % particlePool.size();

    // oob decrement + respawn logic
    if ( glm::any( glm::lessThanEqual( particlePool[ idx ].xyz(), vec3( minExtents - ivec3( 10 ) ) ) ) ||
        glm::any( glm::greaterThanEqual( particlePool[ idx ].xyz(), vec3( maxExtents + ivec3( 10 ) ) ) ) ) {
        // idea is that when you stray outside of the bounding volume, you suffer some attrition...
        particlePool[ idx ].w--;
        if ( particlePool[ idx ].w < 0.0f ) {
            // and eventually when you die, you respawn somewhere on the boundary
            respawnParticle( particlePool[ idx ] );
        }
    }

    // move the particle slightly -> this should be parameterized with "TEMPERATURE"
    rng jitter( -1.0, 1.0f );
    particlePool[ idx ].x += jitter();
    particlePool[ idx ].y += jitter();
    particlePool[ idx ].z += jitter();

    // are we going to bond to something? is there a nearby anchored particle?

    // get the transform of the particle that you want to bond to...

    // find the closest bonding location...

    // and add a particle with the indicated transform
        // right now we will use only position, but orientation is important for crystal lattice
    const mat4 pTransform = glm::translate( mat4( 1.0f ), particlePool[ idx ].xyz() );
    anchorParticle( particlePool[ idx ].xyz(), pTransform );
}

//=================================================================================================
// for dispatching particle updates
atomic_uintmax_t jobCounter;

// threadpool setup
constexpr int NUM_THREADS = 69;
bool threadFences[ NUM_THREADS ] = { true };
bool threadKill = false;
std::thread threads[ NUM_THREADS ];
//=================================================================================================
#include "reporter.h" // proc filesystem reading
//=================================================================================================
int main () {
    // an initial point in the model, so we have something to bond to
    anchorParticle( vec3( 0.0f ), mat4( 1.0f ) );

    // "service" thread, to keep the proc data updated
    std::thread procUpdaterThread = std::thread(
        [&]() {
            while ( !threadKill ) {    
                updateProcData();
                sleep_for( 5ms );
            }
        }
    );

    // dispatching threads:
	for ( int id = 0; id < NUM_THREADS; id++ ) {
        const int myThreadID = id;
		threads[ myThreadID ] = ( myThreadID == 0 ) ? std::thread(
		   [=] () { // this is the reporter thread
				auto tStart = std::chrono::high_resolution_clock::now();
                size_t update = 0;

                int window_1_left = 20;
                int window_1_top = 10;
                int window_1_width = 40;
                int window_1_height = 20;

                auto window_1 = Window({
                  .inner = DummyWindowContent(),
                  .title = "First window",
                  .left = &window_1_left,
                  .top = &window_1_top,
                  .width = &window_1_width,
                  .height = &window_1_height,
                });

                auto window_2 = Window({
                  .inner = DummyWindowContent(),
                  .title = "My window",
                  .left = 40,
                  .top = 20,
                });

                auto window_3 = Window({
                  .inner = DummyWindowContent(),
                  .title = "My window",
                  .left = 60,
                  .top = 30,
                });

                auto window_4 = Window({
                  .inner = DummyWindowContent(),
                });

                auto window_5 = Window({});

                auto window_container = Container::Stacked({
                  window_1,
                  window_2,
                  window_3,
                  window_4,
                  window_5,
                });

                auto display_win_1 = Renderer([&] {
                    string s;
                    int i = 0;
                    for ( auto& p : usagePercentage ) {
                        s += std::to_string( p ) + ( ( i++ % 8 == 0 ) ? "\n" : " " );
                    }
                    return text( s + "window_1: " +  //
                            std::to_string(window_1_width) + "x" +
                            std::to_string(window_1_height) + " + " +
                            std::to_string(window_1_left) + "," +
                            std::to_string(window_1_top));
                });

                auto layout = Container::Vertical({
                  display_win_1,
                  window_container,
                });

                auto screen = ScreenInteractive::TerminalOutput();
                // screen.Loop(layout);


            // this is another way... but still the windows don't update... hmm.

                ftxui::Loop loop( &screen, layout );

                 // Or in a loop:
                while (!loop.HasQuitted()) {
                   screen.RequestAnimationFrame();
                   loop.RunOnce();

                   sleep_for( 100ms );
                }

/*
                while ( !threadKill ) {
                    // update the proc activity samples initially, so we know how many CPUs... 
                    updateProcData();
                    
                    // once every 100ms ( or whatever ) prepare an FTXUI frame for terminal output
                    if ( update % 100 == 0 ) {

                        // Access a specific pixel at (10, 5)
                        auto& pixel = screen.PixelAt( 10, 5 );

                        // Set properties of the pixel.
                        pixel.character = U'X';
                        pixel.foreground_color = ftxui::Color::Red;
                        pixel.background_color = ftxui::Color::RGB( 0, 255, 0 );
                        pixel.bold = true; // Set bold style
                        screen.Print(); // Print the screen to the terminal

                        // show the maximum extents of the crystal elements
                        // show the number of anchored partices
                        // graphical preview of some sort? not sure
                        // CPU activity graph with the averaged proc activity
                    }

                    // sleep this thread for 1ms... or whatever the proc polling interval is
                    update++;
                    std::this_thread::sleep_for( 1ms );
*/

                cout << "Killing Worker Threads" << endl;
                threadKill = true;

                // save the data out, or whatever

                cout << "Reporter Thread Exiting" << endl;
                return;
            }
		) : std::thread(
			[=] () {
                // this is one of the worker threads...
                thread_local const int myThreadID = id;
                while ( !threadKill ) {
                    // check my fence...
                    if ( threadFences[ myThreadID ] ) { // if I'm operating, hit the jobCounter
                        // do a particle update based on the number returned
                        particleUpdate( jobCounter.fetch_add( 1 ) );
                    } else {
                        // if I'm not, sleep 1ms
                        std::this_thread::sleep_for( 1ms );
                        // cout << "I'm dead " << myThreadID << endl;
                    }
                }
                // cout << "Thread " << id << " Exiting" << endl;
                return;
			}
		);
	}

	for ( auto& thread : threads )
		thread.join();

    procUpdaterThread.join();

    cout << "All threads should be finished..." << endl;
    cout << "sizeof(uintmax_t)=" << sizeof( uintmax_t ) << endl;
    cout << "sizeof(size_t)=" << sizeof( size_t ) << endl;

	return 0;
}
