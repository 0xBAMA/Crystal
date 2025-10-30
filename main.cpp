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
#include "random.h"
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
const float temperature = 3.0f;
thread_local rng jitter( -temperature, temperature );
void particleUpdate ( uintmax_t jobIndex ) {
    uintmax_t idx = jobIndex % particlePool.size();
    vec4 &particle =  particlePool[ idx ];

    // oob decrement + respawn logic
    if ( glm::any( glm::lessThanEqual( particle.xyz(), vec3( minExtents - ivec3( 10 ) ) ) ) ||
        glm::any( glm::greaterThanEqual( particle.xyz(), vec3( maxExtents + ivec3( 10 ) ) ) ) ) {
        // idea is that when you stray outside of the bounding volume, you suffer some attrition...
        particle.w--;
        if ( particle.w < 0.0f ) {
            // and eventually when you die, you respawn somewhere on the boundary
            respawnParticle( particle );
        }
    }

    // move the particle slightly -> this should be parameterized with "TEMPERATURE"
    particle.x += jitter();
    particle.y += jitter();
    particle.z += jitter();

    // are we going to bond to something? is there a nearby anchored particle?

    // get the transform of the particle that you want to bond to...

    // find the closest bonding location...

    // and add a particle with the indicated transform
        // right now we will use only position, but orientation is important for crystal lattice
    const mat4 pTransform = glm::translate( mat4( 1.0f ), particle.xyz() );
    anchorParticle( particle.xyz(), pTransform );
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

                bool checked[3] = {false, false, false};
                float slider = 50;

                auto window_1 = Window({
                    .inner = Container::Vertical({
                        Checkbox("Check me", &checked[0]),
                        Checkbox("Check me", &checked[1]),
                        Checkbox("Check me", &checked[2]),
                        Slider("Slider", &slider, 0.f, 100.f),
                        Renderer( [] () {
                            return vbox({
                                text( "Test" ) | color( Color::RGB( 255, 34, 0 ) ),
                                text( "Test" ) | color( Color::RGB( 127, 127, 127 ) ) | bgcolor( Color::RGB( 34, 0, 0 ) ),
                            });
                        }),
                    }),
                    .title = "Control Window",
                    .left = &window_1_left,
                    .top = &window_1_top,
                    .width = &window_1_width,
                    .height = &window_1_height,
                });

                auto window_2 = Window({
                    .inner = Renderer([] (bool focused) {
                        Elements temp, accum;
                        for ( int j = 0; j < 6; j++ ) {
                            for ( int i = 0; i < 12; i++ ) {
                                int idx = i + 6 * j;
                                temp.push_back( text( " " ) | bgcolor( Color::RGB( 0, 255 * usagePercentage[ idx ], 0 ) ) );
                            }
                            accum.push_back( hbox( temp ) );
                            temp.resize( 0 );
                        }
                        return vbox( accum );
                    }),
                    .title = "CPU Activity",
                    .width = 14,
                    .height = 8,
                });

                auto window_container = Container::Stacked({
                  window_1,
                  window_2,
                });

                auto screen = ScreenInteractive::TerminalOutput();
                ftxui::Loop loop( &screen, window_container );

                // "main loop"
                while (!loop.HasQuitted()) {
                   screen.RequestAnimationFrame();
                   loop.RunOnce();

                   sleep_for( 100ms );
                }

                // signal that all threads should exit
                threadKill = true;

                // save the data out, or whatever

                return;
            }
		) : std::thread(
			[&] () {
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
                    }
                }
                return;
			}
		);
	}

    // join all worker threads back to main
	for ( auto& thread : threads )
		thread.join();

    // join the proc updater thread back to main
    procUpdaterThread.join();

	return 0;
}
