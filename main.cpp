//=================================================================================================
// command line output
#include <string>
#include <iostream>
using std::cout;
using std::endl;
using std::string;
using std::to_string;
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
using std::chrono::high_resolution_clock;
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
using std::make_shared;
using std::shared_ptr;
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

// constants for shortening calculations of points + vector offsets
constexpr vec4 p0 = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
constexpr vec4 v0 = vec4( 0.0f, 0.0f, 0.0f, 0.0f );

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
    std::atomic< std::int32_t > count = { 0 };
    std::vector< mat4 > particles;

    gridCell() {
        count.store( 0 );
        particles.resize( 0 );
        particles.reserve( 128 );
    }

    int32_t GetCount () {
        // this should be able to rely on the count...
        // return std::min( int( anchoredParticles[ p ].count.load() ),
                          // int( anchoredParticles[ p ].particles.size() ) );
        return count;
    }

    void Add( const mat4 pTransform ) {
        // add it to the list... mutex will be locked at this time
        count++;
        particles.push_back( pTransform );
    }

    mat4 Get( int idx ) {
        // get index i from the list... we need to avoid seg faulting, so return a dummy value if we don't have it
        if ( count == 0 || idx >= count ) {
            return mat4( 1.0f );
        } else {
            return particles[ idx ];
        }
    }
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
void respawnParticle ( vec4 &p ) {
    thread_local rngi pick( 0, 2 );
    thread_local rng pick2( 0.0f, 1.0f );

    thread_local bool face = ( pick2() < 0.5f );
    constexpr float margin = 10.0f;
    p.x = glm::mix( minExtents.x - margin, maxExtents.x + margin, pick2() );
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

void anchorParticle ( const ivec3 iP, const mat4 &pTransform ) {
    // we need to anchor this particle... but first, lock the mutex so only one thread can do this at once
    lock_guard< mutex > lock( anchoredParticlesGuard ); // mutex object entering scope locks... declaration is blocking 

    // push the constructed mat4 onto the list
    anchoredParticles[ iP ].Add( pTransform );

    // tracking how many particles have been added, and the min and max extents
    minExtents = glm::min( iP, minExtents );
    maxExtents = glm::max( iP, maxExtents );
    numAnchored++;

    // and the mutex leaves scope, unblocking for the next write access
}

// this is the update, operating on a particular particle
const float temperature = 3.0f; // make this a sim parameter, eventually
thread_local rngN jitter( 0.0f, 0.1f );
void particleUpdate ( uintmax_t jobIndex ) {
    thread_local const uintmax_t idx = jobIndex % particlePool.size();
    thread_local vec4 &particle = particlePool[ idx ];

    cout << "Starting Update at " << to_string( particle ) << endl;

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

    // move the particle slightly
    particle.x += temperature * jitter();
    particle.y += temperature * jitter();
    particle.z += temperature * jitter();

    // are we going to bond to something? is there a nearby anchored particle?
    // first, let's look at all the particles in the local neighborhood...
    thread_local std::vector< mat4 > nearbyPoints; // eventually this should be a static scratch pool per thread to avoid allocations
    for ( int x = -1; x <= 1; x++ )
    for ( int y = -1; y <= 1; y++ )
    for ( int z = -1; z <= 1; z++ ) {
        thread_local const ivec3 loc = ivec3( particle.xyz() ) + ivec3( x, y, z );
        thread_local gridCell &gc = anchoredParticles[ loc ];
        thread_local const int count = gc.GetCount();
        if ( count > 0 ) {
            for ( thread_local int i = 0; i < count; i++ ) {
                cout << "attempting read at " << to_string( loc ) << " " << i << " / " << count << endl;
                thread_local mat4 mat = gc.Get( i );
                nearbyPoints.push_back( mat );
            }
        }
    }

    // we need to find out which, if any, of these points can be anchored to...
    if ( nearbyPoints.size() != 0 ) {

        // cout << "I have Neighbors!!" << endl;

        // finding the closest point... we know we have 1, so we start with that
        thread_local mat4 closestPointTransform = nearbyPoints[ 0 ];
        thread_local float closestPointDistance = glm::distance(
            particle.xyz(), ( closestPointTransform * p0 ).xyz()
        );

        // if we have more than one point to consider, compare them
        for ( auto& transform : nearbyPoints ) {
            const float d = glm::distance(
                particle.xyz(), ( transform * p0 ).xyz()
            );
            if ( d < closestPointDistance ) {
                closestPointDistance = d;
                closestPointTransform = transform;
            }
        }

        // some additional bonding criteria...?
        if ( true ) { // close enough... random hash... etc
            // figure out which of bonding sites you want to bond to... probably the closest one of them

            // the mat4 tells us the orientation and the position of the point

            // and add a particle with the indicated transform
                // right now we will use only position, but orientation is important for crystal lattice
            thread_local const mat4 pTransform = glm::translate( mat4( 1.0f ), particle.xyz() );

            // mutex is locked, only during add... math happens outside, nice
            anchorParticle( ivec3( particle.xyz() ), pTransform );

            // since we bound to a surface, the point location is no longer valid and should not proceed
            respawnParticle( particle ); // we will now respawn somewhere on the sim boundary
        }
    }

}

//=================================================================================================
// for dispatching particle updates
atomic_uintmax_t jobCounter { 0 };

// threadpool setup
constexpr int NUM_THREADS = 2;
bool threadFences[ NUM_THREADS ];
bool threadKill;
std::thread threads[ NUM_THREADS ];
//=================================================================================================
#include "reporter.h" // proc filesystem reading
//=================================================================================================
int main () {
    // pump initial proc data
    updateProcData();
    updateProcData();

    // setting initial program state
    threadKill = false;
    // set all the thread fences "true"
    // cout << "Enable Worker Threads............. ";
    for ( auto& fence : threadFences )
        fence = true;
    // cout << "Done." << endl;
    
    cout << "Spawning Reporter Thread.......... ";
	std::thread reporterThread = std::thread(
        [&] () { // this is the reporter thread
			auto tStart = high_resolution_clock::now();
            cout << "Done." << endl;

            size_t update = 0;

            int window_1_left = 10;
            int window_1_top = 10;
            int window_1_width = 40;
            int window_1_height = 20;

            bool checked[3] = {false, false, false};
            float slider = 50;

            auto window_1 = Window({
                .inner = Container::Vertical({
                    // Checkbox("Check me", &checked[0]),
                    // Checkbox("Check me", &checked[1]),
                    // Checkbox("Check me", &checked[2]),
                    // Slider("Slider", &slider, 0.f, 100.f),
                    Renderer( [&] ( bool focused ) {
                        auto c1 = color( Color::RGB( 255, 34, 0 ) );
                        auto c2 = color( Color::RGB( 255, 255, 34 ) );
                        return vbox({
                            hbox({ text( "Job Counter:         " ) | c1, text( std::to_string( jobCounter.load() ) ) | c2 }),
                            hbox({ text( "Anchored Particles:  " ) | c1, text( std::to_string( numAnchored ) ) | c2 }),
                            hbox({ text( "Extents:" ) | c1 }),
                            hbox({ text( " x:  " ) | c1, text( to_string( minExtents.x ) + " " ) | c2, text( to_string( maxExtents.x ) ) | c2 }),
                            hbox({ text( " y:  " ) | c1, text( to_string( minExtents.y ) + " " ) | c2, text( to_string( maxExtents.y ) ) | c2 }),
                            hbox({ text( " z:  " ) | c1, text( to_string( minExtents.z ) + " " ) | c2, text( to_string( maxExtents.z ) ) | c2 }),
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
                            float intensity = usagePercentage[ idx ];
                            temp.push_back( text( " " ) | bgcolor( Color::RGB( 255 * sqrt( intensity ), 128 * intensity, 0 ) ) );
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

            auto screen = ScreenInteractive::FixedSize( 80, 30 );
            auto ftxDAG = CatchEvent( window_container, [&]( Event event ) {
                if ( event == Event::Character('q') ) {
                    screen.ExitLoopClosure()();
                    return true;
                }
                return false;
            });

            // an initial point in the model, so we have something to bond to
            // cout << "Anchoring Initial Seed Particles.. ";
            rng pR( -2.0f, 2.0f );
            for ( int i = 0; i < 10; i++ ) {
                mat4 transform = glm::translate( glm::rotate( mat4( 1.0f ), pR(), normalize( vec3( pR(), pR(), pR() ) ) ), 10.0f * vec3( pR(), pR(), pR() ) );
                anchorParticle( transform * p0, transform );
            }
            // cout << "Done." << endl;

            // int detectedPointCount = 0;
            // for ( auto& [k,v] : anchoredParticles )
                // for ( auto& p : v.particles ) {
                    // detectedPointCount++;
                    // cout << "found: " << to_string( p ) << " at " << to_string( k ) << endl;
                    // cout << "confirm: " << to_string( p * vec4( 0.0f, 0.0f, 0.0f, 1.0f ) ) << endl;
                // }
            // cout << "Confirm Contents: map has " << detectedPointCount << " elements" << endl;

            // we need to make sure there are particles to update...
            // cout << "Spawning Particles................ ";
            particlePool.resize( 100000 );
            for ( auto& p : particlePool ) {
                respawnParticle( p );
            }
            // cout << "Done." << endl;

            // "main loop"
            // cout << "Entering Main Loop..." << endl;
            // ftxui::Loop loop( &screen, ftxDAG );
            // bool timeout = ( float( std::chrono::duration< float, std::milli >( high_resolution_clock::now() - tStart ).count() ) > 1000.0f );
            // while (!loop.HasQuitted() && !timeout ) {
               // screen.RequestAnimationFrame();
               // loop.RunOnce();
               // sleep_for( 100ms );
            // }

           sleep_for( 1000ms );

            // signal that all threads should exit
            threadKill = true;

            // save the data out, or whatever

            return;
        }
	);

    // touch some chunk of the hashmap to preallocate?
    // cout << "hashmap preallocate... ";
    // mat4 temp;
    // for ( int x = -10; x < 10; x++ ) {
    // for ( int y = -10; y < 10; y++ ) {
    // for ( int z = -10; z < 10; z++ ) {
        // temp = readParticle( ivec3( x, y, z ), 0 );
    // }}}
    // ( void ) temp;
    // cout << "finished." << endl;

    // "service" thread, to keep the proc data updated
    cout << "Spawning Proc Updater Thread...... ";
    std::thread procUpdaterThread = std::thread(
        [&]() {
            while ( !threadKill ) {    
                updateProcData();
                sleep_for( 5ms );
            }
        }
    );
    cout << "Done." << endl;

    // dispatching threads:
    cout << "Dispatching Worker Threads (" << NUM_THREADS << ")... ";
	for ( int id = 0; id < NUM_THREADS; id++ ) {
		threads[ id ] = std::thread(
			[&] () {
                // this is one of the worker threads...
                thread_local int myThreadID = id;
                cout << "thread " << myThreadID << " init" << endl;
                sleep_for( 100ms );
                while ( !threadKill ) {
                    // check my fence...
                    if ( threadFences[ myThreadID ] ) { // if I'm operating, hit the jobCounter
                        // do a particle update based on the number returned
                        cout << "thread " << myThreadID << " attempting update" << endl;
                        particleUpdate( jobCounter.fetch_add( 1 ) );
                    } else {
                        // if I'm not, sleep 1ms
                        std::this_thread::sleep_for( 1ms );
                    }
                }
                cout << "thread " << myThreadID << " leaving scope" << endl;
                return;
			}
		);
	}
	cout << "Done." << endl;

    
    // join all worker threads back to main
	for ( auto& thread : threads )
		thread.join();

    // join the proc updater thread back to main
    procUpdaterThread.join();

    // join the reporter thread, now that everything else has terminated
    reporterThread.join();
    cout << "Terminating....................... Done." << endl;

	return 0;
}
