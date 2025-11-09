//=================================================================================================
// command line output
#include <string>
#include <iostream>
#include <sstream>
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
// std::clamp
#include <algorithm>
using std::clamp;
using std::max;
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
using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat3;
using glm::mat4;
using glm::ivec2;
using glm::pi;
using glm::ivec3;
using glm::dot;

// constants for shortening calculations of points + vector offsets
constexpr vec4 p0 = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
constexpr vec4 v0 = vec4( 0.0f, 0.0f, 0.0f, 0.0f );
constexpr vec4 vX = vec4( 1.0f, 0.0f, 0.0f, 0.0f );
constexpr vec4 vY = vec4( 0.0f, 1.0f, 0.0f, 0.0f );
constexpr vec4 vZ = vec4( 0.0f, 0.0f, 1.0f, 0.0f );
constexpr mat4 identity = mat4( 1.0f );

// key hash needed for std::unordered_map with ivec3 keys
namespace std {
	template<> struct hash< ivec3 > {
		// custom specialization of std::hash can be injected in namespace std
		std::size_t operator()( ivec3 const& s ) const noexcept {
			std::size_t h1 = std::hash< int >{}( s.x );
			std::size_t h2 = std::hash< int >{}( s.y );
			std::size_t h3 = std::hash< int >{}( s.z );
			return h1 ^ ( h2 << 12 ) ^ ( h3 << 24 );
		}
	};
}
//=================================================================================================
// random number generation utilities
#include "random.h"
//===== STB ===========================================================================================================
// Sean Barrett's public domain load, save, resize libs - need corresponding define in the ./stb/impl.cc file,
	// before their inclusion, which is done by the time compilation hits this point - they can be straight
	// included, here, as follows:
#include "stb/stb_image.h"          // https://github.com/nothings/stb/blob/master/stb_image.h
#include "stb/stb_image_write.h"	// https://github.com/nothings/stb/blob/master/stb_image_write.h
#include "stb/stb_image_resize.h"	// https://github.com/nothings/stb/blob/master/stb_image_resize.h
//=================================================================================================
#include "hashMap/inc/HashMap.h"
//=================================================================================================
#include "gif.h" // gif output lib - https://github.com/charlietangora/gif-h
/* Basic Usage:

#include <vector>
#include <cstdint>
#include <gif.h>
int main()
{
	int width = 100;
	int height = 200;
	std::vector<uint8_t> black(width * height * 4, 0);
	std::vector<uint8_t> white(width * height * 4, 255);

	auto fileName = "bwgif.gif";
	int delay = 100;
	GifWriter g;
	GifBegin(&g, fileName, width, height, delay);
	GifWriteFrame(&g, black.data(), width, height, delay);
	GifWriteFrame(&g, white.data(), width, height, delay);
	GifEnd(&g);

	return 0;
} */
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

float remap ( float in, float aL, float aH, float bL, float bH ) {
    return bL + ( in - aL ) * ( bH - bL ) / ( aH - aL );
}

struct gridCell {
    vector< mat4 > particles;
    std::shared_mutex mutex;

    gridCell() {
        particles.resize( 0 );
        particles.reserve( 16 );
    }

    size_t GetCount () {
        // this is the non-exclusive read mutex
        std::shared_lock lock( mutex );
        return particles.size();
    }

    void Add( const mat4 &pTransform ) {
        // add it to the list... exclusive mutex is locked at this time
        std::unique_lock lock( mutex );
        if ( particles.size() < 128 )
            particles.push_back( pTransform );
    }

    mat4 Get( const int &idx ) {
        // locking the non-exclusive read mutex
        std::shared_lock lock( mutex );
        return particles[ idx ];
    }
};

// the list of particles which have been anchored
CTSL::HashMap< ivec3, std::shared_ptr< gridCell > > anchoredParticles;
// mutex anchoredParticlesGuard;

// maintaining stats on the above container
ivec3 minExtents = ivec3( -1 );
ivec3 maxExtents = ivec3( 1 );
int numAnchored = 0;

// get a point on the boundary
thread_local rng pick( 0.0f, 1.0f );
void respawnParticle ( vec4 &p ) {
    const bool face = ( pick() < 0.5f );
    constexpr float margin = 0.0f;
    p.x = glm::mix( minExtents.x - margin, maxExtents.x + margin, pick() );
    p.y = glm::mix( minExtents.y - margin, maxExtents.y + margin, pick() );
    p.z = glm::mix( minExtents.z - margin, maxExtents.z + margin, pick() );

    switch ( int( std::floor( pick() * 3.01f ) ) ) {
        // we will be on one of the parallel x faces, flatten
        case 0: p.x = face ? minExtents.x - margin : maxExtents.x + margin; break;

        // ditto, y faces
        case 1: p.y = face ? minExtents.y - margin : maxExtents.y + margin; break;

        // z faces
        case 2: p.z = face ? minExtents.z - margin : maxExtents.z + margin; break;

        // won't hit much, but will be uniform random spawn if you do
        default:
            break;
    }

    // the .w will be a counter... it is decremented any update during which it is outide the current bounding volume
    p.w = 100.0f; // I think this is a nice mechanism - it can wander, but in a bounded way - when it hits zero, respawn
}

void anchorParticle ( const ivec3 iP, const mat4 &pTransform ) {
    // we need to anchor this particle... protect behind a mutex
    // lock_guard< mutex > lock( anchoredParticlesGuard );

    // push the constructed mat4 onto the list
    if ( shared_ptr< gridCell > gcp; anchoredParticles.find( iP, gcp ) ) {
        // we have a gridCell object already instantiated for managing the contents...
        gcp->Add( pTransform );
    } else {
        // need to insert a new key... init the gridCell and add the mat4
        shared_ptr< gridCell > temp = make_shared< gridCell >();
        anchoredParticles.insert( iP, temp );
        temp->Add( pTransform );
    }

    // tracking how many particles have been added, and the min and max extents
    minExtents = glm::min( iP, minExtents );
    maxExtents = glm::max( iP, maxExtents );
    numAnchored++;

    // and the mutex leaves scope, unblocking for the next write access
}

// this is the update, operating on a particular particle
float temperature = 5.0f;
thread_local rngN jitter( 0.0f, 0.1f );

// these are the bonding sites
vector< vec4 > bondingSiteOffsets;

// the particles are the diffusion-limit mechanism
// constexpr int32_t NUM_PARTICLES = 10'000'000;
constexpr int32_t NUM_PARTICLES = 1'000;

vec4 particlePool[ NUM_PARTICLES ];
void particleUpdate ( uintmax_t jobIndex ) {
    // thread_local const uintmax_t idx = jobIndex % NUM_PARTICLES;
    const uintmax_t idx = jobIndex % NUM_PARTICLES;
    vec4 &particle = std::ref( particlePool[ idx ] );

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
    std::vector< mat4 > nearbyPoints;

    // looking at the local neighborhood...
    const auto& s = { -1, 0, 1 };
    for ( int x : s )
    for ( int y : s )
    for ( int z : s ) {
        // if ( x == 0 && y == 0 && z == 0 ) continue;
        const ivec3 loc = ivec3( particle.xyz() ) + ivec3( x, y, z );

        // "find()" will also return the pointer to the container contents
        if ( std::shared_ptr< gridCell > gcp; anchoredParticles.find( loc, gcp ) ) {
            // we have located a cell which contains particles...
            const size_t count = gcp->GetCount();
            for ( int i = 0; i < count; i++ ) {
                nearbyPoints.push_back( gcp->Get( i ) );
            }
        } // else we did not find any contents here
    }

    // we need to find out which, if any, of these points can be anchored to...
    if ( !nearbyPoints.empty() ) {

        // finding the closest point... we know we have 1, so we start with that
        mat4 closestPointTransform = nearbyPoints[ 0 ];
        vec3 closestPointTransformed = ( closestPointTransform * p0 ).xyz();
        float closestPointDistance = glm::distance(
            particle.xyz(), closestPointTransformed
        );

        // if we have more than one point to consider, compare them
        for ( auto& transform : nearbyPoints ) {
            const vec3 p = ( transform * p0 ).xyz();
            const float d = glm::distance( particle.xyz(), p );
            if ( d < closestPointDistance ) {
                closestPointDistance = d;
                closestPointTransform = transform;
                closestPointTransformed = p;
            }
        }

        // some additional bonding criteria...?
        if ( closestPointDistance < 1.5f ) { // close enough... random hash... etc
            // figure out which of bonding sites you want to bond to... probably the closest one of them
            vec3 closestBondingPointOffset;
            float closestBondingPointDistance = 10000.0f;
            
            // now looking at a list, find the closest...
            // cout << to_string( closestPointTransform ) << endl;
#if 1
            for ( auto& bpo : bondingSiteOffsets ) {
                vec4 transformedBondingPointOffset = closestPointTransform * bpo;
                const float d = glm::distance( particle.xyz(), closestPointTransformed + transformedBondingPointOffset.xyz() );
                if ( d < closestBondingPointDistance ) {
                    closestBondingPointDistance = d;
                    closestBondingPointOffset = transformedBondingPointOffset.xyz();
                }
            }
#else
            // just pick one at random...
            closestBondingPointOffset = closestPointTransform * bondingSiteOffsets[ clamp( size_t( floor( pick() * bondingSiteOffsets.size() ) ), size_t( 0 ), bondingSiteOffsets.size() ) ];
#endif
            // apply the offset to the particle, and the orientation from before....

            // the mat4 tells us the orientation and the position of the point
            // we have a very low chance to alter the orientation... jitter position, etc
            if ( pick() < 0.01f ) {
                closestBondingPointOffset += vec3( jitter(), jitter(), jitter() );
                closestPointTransform = glm::translate( glm::rotate( glm::translate( closestPointTransform, -closestPointTransformed ), jitter(), glm::normalize( vec3( jitter(), jitter(), jitter() ) ) ), closestPointTransformed );
            }
            
            const vec4 TvX = closestPointTransform * vX;
            const vec4 TvY = closestPointTransform * vY;
            const vec4 TvZ = closestPointTransform * vZ;
            
            const vec3 p = closestPointTransformed + closestBondingPointOffset;
            const mat4 pTransform = mat4( TvX, TvY, TvZ, vec4( p, 1.0f ) );

            // mutex is locked, only during add... math happens outside, nice
            anchorParticle( ivec3( p ), pTransform );

            // since we bound to a surface, the point location is no longer valid and should not proceed
            respawnParticle( particle ); // we will now respawn somewhere on the sim boundary
        }
    }
}
//=================================================================================================
void prepareOutput() {
       // create a linear buffer of all the point locations in minimum representation
    size_t maxSize = 0;
    vector< vec3 > points;
    points.reserve( numAnchored );
    {
        // lock_guard< mutex > lock( anchoredParticlesGuard );
        for ( int x = minExtents.x; x < maxExtents.x; x++ ) {
            for ( int y = minExtents.y; y < maxExtents.y; y++ ) {
                for ( int z = minExtents.z; z < maxExtents.z; z++ ) {
                    if ( shared_ptr< gridCell > gcp; anchoredParticles.find( ivec3( x, y, z ), gcp ) ) {
                        maxSize = max( maxSize, gcp->particles.size() );
                        for ( auto& p : gcp->particles ) {
                            vec3 pT = ( p * p0 ).xyz();
                            points.push_back( pT );
                        }
                    }
                }
            }
        }
    }


    // information about anchored particles
    int binCountsA[ 1000 * 1000 ];
    float binHeights[ 1000 * 1000 ];

    int maxCountA = 0;
    int nonzeroBinsA = 0;
    for ( auto& b : binCountsA ) { b = 0; }
    for ( auto& p : points ) {
        vec3 loc = vec3(
            clamp( remap( p.x, minExtents.x, maxExtents.x, 0.0f, 1000.0f ), 0.0f, 999.0f ),
            clamp( remap( p.y, minExtents.y, maxExtents.y, 0.0f, 1000.0f ), 0.0f, 999.0f ),
            clamp( remap( p.z, minExtents.z, maxExtents.z, 0.0f, 1000.0f ), 0.0f, 999.0f )
        );
        const int idx = int( loc.x ) + 1000 * int( loc.y ); 
        binCountsA[ idx ]++;
        binHeights[ idx ] = max( loc.z, binHeights[ idx ] );
        maxCountA = max( maxCountA, binCountsA[ idx ] );
    }

    cout << "processed " << points.size() << " particles" << endl;
    cout << "found max " << maxCountA << " points per bin" << endl;
    // write out the image
    std::vector< uint8_t > data;
    for ( int i = 0; i < ( 1000 * 1000 ); i++ ) {
        // some height term and a density term
        float h = binHeights[ i ] / 1000.0f;
        // float b = ( binCountsA[ i ] == 0 ) ? 0 : ( 255 * glm::exp( -0.001f * float( binCountsA[ i ] ) / float( maxCountA ) ) );
        float b = 255 * glm::pow( float( binCountsA[ i ] )  / float( maxCountA ), 0.8f );        

        vec3 c = glm::mix( vec3( 1.0f ), vec3( 0.0f, 1.0f, 0.3f ), vec3( h ) );

        data.push_back( b * c.x );
        data.push_back( b * c.y );
        data.push_back( b * c.z );
        
        data.push_back( 255 );
    }

	auto now = std::chrono::system_clock::now();
	auto inTime_t = std::chrono::system_clock::to_time_t( now );
	std::stringstream ssA;
	ssA << std::put_time( std::localtime( &inTime_t ), "Crystal-%Y-%m-%d at %H-%M-%S.png" );
    stbi_write_png( ssA.str().c_str(), 1000, 1000, 4, &data[ 0 ], 4000 );
}
//=================================================================================================
// for dispatching particle updates
atomic_uintmax_t jobCounter { 0 };

// threadpool setup
constexpr int NUM_THREADS = 72;
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
    for ( auto& fence : threadFences )
        fence = true;

    // we need to make sure there are particles to update...
    // cout << "Spawning Particles................ ";
    for ( auto& p : particlePool ) {
        // respawnParticle( p );
        p.x = remap( pick(), 0.0f, 1.0f, -5.0f, 5.0f );
        p.y = remap( pick(), 0.0f, 1.0f, -5.0f, 5.0f );
        p.z = remap( pick(), 0.0f, 1.0f, -4.0f, 4.0f );
        p.w = 100.0f;
    }
    // cout << "Done." << endl;

    // an initial point in the model, so we have something to bond to
    // cout << "Anchoring Initial Seed Particles.. ";
    rng pR( -2.0f, 2.0f );
    for ( int i = 0; i < 10; i++ ) {
        mat4 transform = glm::rotate( glm::translate( identity, 10.0f * vec3( 2.0f * pick(), 2.0f * pick(), 0.1f * jitter() ) ), 10.0f * pR(), glm::normalize( vec3( jitter(), jitter(), jitter() ) ) );
        anchorParticle( transform * p0, transform );
    }
    // cout << "Done." << endl;

    // we need to pick a set of offsets to use for the crystal growth...
        // start with a "tetragonal", which will emphasize orientation because it has longer offsets along one axis
    // the angle is uniform, so we will leave this as just a nonuniformly scaled basis...
    const float xXx = 0.3f;
    const float yYy = 0.3f;
    const float zZz = 0.1618f;

    const mat4 rX = glm::rotate( identity, -3.1415926535f / 3.0f, vec3( 0.0f, 1.0f, 0.0f ) );
    const mat4 rY = identity;
    const mat4 rZ = identity;

    // note that 6 bonding points is in no way a constraint here
    bondingSiteOffsets = {
        // rX * vec4( 0.0f, 0.0f, -zZz, 0.0f ),
        rX * vec4( 0.0f, 0.0f, zZz, 0.0f ),
        rX * rX * vec4( 0.0f, 0.0f, zZz, 0.0f ),
        rX * rX * rX * vec4( 0.0f, 0.0f, zZz, 0.0f ),
        // rY * vec4( 0.0f, -yYy, 0.0f, 0.0f ),
        rY * vec4( 0.0f, yYy, 0.0f, 0.0f ),
        // rZ * vec4( -xXx, 0.0f, 0.0f, 0.0f ),
        rZ * vec4( xXx, 0.0f, 0.0f, 0.0f ),
    };
    
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
                    Slider( "Temperature", &temperature, 0.0f, 40.0f ),
                    Renderer( [&] ( bool focused ) {
                        auto c1 = color( Color::RGB( 255, 34, 0 ) );
                        auto c2 = color( Color::RGB( 255, 255, 34 ) );
                        return vbox({
                            hbox({ text( "Uptime:              " ) | c1, text( std::to_string( float( std::chrono::duration< float, std::milli >( high_resolution_clock::now() - tStart ).count() ) / 1000.0f ) ) | c2, text( "s" ) | c1 }),
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
                    // save out the gif
                    // GifEnd( &g );
                    screen.ExitLoopClosure()();
                    return true;
                }

                if ( event == Event::Character('w') ) {
                    prepareOutputScreenshot();
                    return true;
                }

                return false;
            });

        // "main loop"
            cout << "Entering Main Loop..." << endl;
            ftxui::Loop loop( &screen, ftxDAG );
            bool quit = false;
            while ( !loop.HasQuitted() && !quit ) {
            // while ( !quit ) {
               screen.RequestAnimationFrame();
               loop.RunOnce();
               sleep_for( 100ms );

               static uintmax_t lastObserved = 0;
               static uintmax_t lastObservedAccum = 0;
               lastObservedAccum += numAnchored - lastObserved;
               lastObserved = numAnchored;
               if ( lastObservedAccum > 5000 ) {
                   prepareOutputGIFFrame( &g );
                   lastObservedAccum = 0;
                   if ( gifFrame > 600 || points.size() > 15000000 ) {
                       quit = true;
                   }
                   cout << "finished " << gifFrame << endl;
               }

               // terminate condition:
               // if (  ) {
               // }
            }

            // signal that all threads should exit and wait for them to do so
            threadKill = true;
            sleep_for( 100ms );

            return;
        }
	);

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
                sleep_for( 100ms );
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
	cout << "Done." << endl;

    // join all worker threads back to main
	for ( auto& thread : threads )
		thread.join();

    // join the proc updater thread back to main
    procUpdaterThread.join();

    // join the reporter thread, now that everything else has terminated
    reporterThread.join();
    cout << "Terminating....................... Done." << endl;

    // save the data out, bake out a preview or whatever for now...
    int outputRotationSteps = 400;
    for ( int i = 0; i < outputRotationSteps; i++ ) {
        cout << "prepping output rotation: " << i << " / " << outputRotationSteps << endl;
        prepareOutputGIFFrameNoPrep( &g );
    }
    GifEnd( &g );
    prepareOutputScreenshot();

	return 0;
}
