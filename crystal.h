//=================================================================================================
// basic IO
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
using std::cout;
using std::endl;
using std::string;
using std::to_string;
//=================================================================================================
// Terminal UI output
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>

#include <ftxui/component/loop.hpp>
#include <ftxui/component/captured_mouse.hpp>       // for ftxui
#include <ftxui/component/component.hpp>            // for Checkbox, Renderer, Vertical
#include <ftxui/component/component_base.hpp>       // for ComponentBase
#include <ftxui/component/screen_interactive.hpp>   // for ScreenInteractive
using namespace ftxui;
//=================================================================================================
// basic containers
#include <vector>
using std::vector;
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
using std::shared_mutex;
using std::lock_guard;
using std::shared_lock;
using std::unique_lock;
//=================================================================================================
// std::clamp
#include <algorithm>
using std::clamp;
using std::max;
//=================================================================================================
// specific width types + atomics, for "program counter"
#include <cstdint>
#include <atomic>
using std::uintmax_t;
using std::atomic_uintmax_t;
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
			return h1 ^ ( h2 << 10 ) ^ ( h3 << 20 );
		}
	};
}
//=================================================================================================
bool IntersectAABB ( const vec3 rO, const vec3 rD, const vec3 min, const vec3 max, float &tMin, float &tMax ) {
    // Intersect() code adapted from:
    //    Amy Williams, Steve Barrus, R. Keith Morley, and Peter Shirley
    //    "An Efficient and Robust Ray-Box Intersection Algorithm"
    //    Journal of graphics tools, 10(1):49-54, 2005
    constexpr float minDistance = -10000.0;
    constexpr float maxDistance =  10000.0;
    int s[ 3 ]; // sign toggle
    // inverse of ray direction
    const vec3 iD = vec3( 1.0 ) / rD;
    s[ 0 ] = ( iD[ 0 ] < 0 ) ? 1 : 0;
    s[ 1 ] = ( iD[ 1 ] < 0 ) ? 1 : 0;
    s[ 2 ] = ( iD[ 2 ] < 0 ) ? 1 : 0;
    const vec3 b[ 2 ] = { min, max }; // bounds
    tMin = ( b[ s[ 0 ] ][ 0 ] - rO[ 0 ] ) * iD[ 0 ];
    tMax = ( b[ 1 - s[ 0 ] ][ 0 ] - rO[ 0 ] ) * iD[ 0 ];
    const float tYMin = ( b[ s[ 1 ] ][ 1 ] - rO[ 1 ] ) * iD[ 1 ];
    const float tYMax = ( b[ 1 - s[ 1 ] ][ 1 ] - rO[ 1 ] ) * iD[ 1 ];
    if ( ( tMin > tYMax ) || ( tYMin > tMax ) ) return false;
    if ( tYMin > tMin ) tMin = tYMin;
    if ( tYMax < tMax ) tMax = tYMax;
    const float tZMin = ( b[ s[ 2 ] ][ 2 ] - rO[ 2 ] ) * iD[ 2 ];
    const float tZMax = ( b[ 1 - s[ 2 ] ][ 2 ] - rO[ 2 ] ) * iD[ 2 ];
    if ( ( tMin > tZMax ) || ( tZMin > tMax ) ) return false;
    if ( tZMin > tMin ) tMin = tZMin;
    if ( tZMax < tMax ) tMax = tZMax;
    return ( ( tMin < maxDistance ) && ( tMax > minDistance ) );
}
//=================================================================================================
float remap ( float in, float aL, float aH, float bL, float bH ) {
    return bL + ( in - aL ) * ( bH - bL ) / ( aH - aL );
}
//=================================================================================================
// some color ramp functions from https://www.shadertoy.com/view/Nd3fR2
vec3 turbo(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((0.192919+t*(1.618437+t*(-39.426098+t*(737.420549+t*(-6489.216487+t*(28921.755478+t*(-72384.553891+t*(107076.097978+t*(-93276.212113+t*(44337.286143+t*-8884.508085)))))))))),
                      (0.101988+t*(1.859131+t*(7.108520+t*(-20.179546+t*11.147684)))),
                      (0.253316+t*(4.858570+t*(55.191710+t*(-803.379980+t*(4477.461997+t*(-14496.039745+t*(28438.311669+t*(-32796.884355+t*(20328.068712+t*-5210.826342)))))))))), 0.0, 1.0);
}

vec3 bone(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((-0.011603+t*(1.066867+t*(-0.673604+t*0.623930))),
                      (0.018446+t*(0.524946+t*(1.185768+t*-0.732596))),
                      (0.004421+t*(1.254048+t*-0.275214))), 0.0, 1.0);
}

vec3 afmhot(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((-0.000000+t*2.000000),
                      (-0.500000+t*2.000000),
                      (-1.000000+t*2.000000)), 0.0, 1.0);
}

vec3 gist_heat(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((0.000000+t*1.500000),
                      (-1.000000+t*2.000000),
                      (-3.000000+t*4.000000)), 0.0, 1.0);
}

vec3 plasma(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((0.057526+t*(2.058166+t*-1.141244)),
                      (-0.183275+t*(0.668964+t*0.479353)),
                      (0.525210+t*(1.351117+t*(-4.013494+t*2.284066)))), 0.0, 1.0);
}

vec3 inferno(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((-0.015449+t*(0.816640+t*(3.399179+t*(-4.796465+t*1.530683)))),
                      (0.000619+t*(0.450682+t*(-1.556978+t*(3.904984+t*-1.764423)))),
                      (0.019123+t*(0.792737+t*(29.365333+t*(-210.608893+t*(622.120191+t*(-942.393021+t*(711.115854+t*-209.780428)))))))), 0.0, 1.0);
}

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((-0.023114+t*(0.883412+t*(2.280390+t*-2.164009))),
                      (-0.000931+t*(0.700294+t*(-3.639731+t*(14.399222+t*(-28.183967+t*(29.245012+t*-11.549071)))))),
                      (0.011971+t*(1.223232+t*(17.782054+t*(-111.294284+t*(282.340184+t*(-384.394777+t*(275.310307+t*-80.251736)))))))), 0.0, 1.0);
}
//=================================================================================================
// random number generation utilities
#include "random.h"
//===== STB =======================================================================================
// Sean Barrett's public domain load, save, resize libs - need corresponding define in the ./stb/impl.cc file,
	// before their inclusion, which is done by the time compilation hits this point - they can be straight
	// included, here, as follows:
#include "stb/stb_image.h"          // https://github.com/nothings/stb/blob/master/stb_image.h
#include "stb/stb_image_write.h"	// https://github.com/nothings/stb/blob/master/stb_image_write.h
#include "stb/stb_image_resize.h"	// https://github.com/nothings/stb/blob/master/stb_image_resize.h
//=================================================================================================
#include "hashMap/inc/HashMap.h"
using CTSL::HashMap;
//=================================================================================================
#include "gif.h" // gif output lib - https://github.com/charlietangora/gif-h
/* Basic Usage:
#include <vector>
#include <cstdint>
#include <gif.h>
int main() {
	int width = 100;
	int height = 200;
	std::vector< uint8_t > black( width * height * 4, 0 );
	std::vector< uint8_t > white( width * height * 4, 255 );

	auto fileName = "bwgif.gif";
	int delay = 100; // in 100th's of a second, not strictly enforced
	GifWriter g;
	GifBegin( &g, fileName, width, height, delay );
	GifWriteFrame( &g, black.data(), width, height, delay );
	GifWriteFrame( &g, white.data(), width, height, delay );
	GifEnd( &g );

	return 0;
} */
//=================================================================================================
struct CrystalSimConfig {

    // todo

    // also render config like colors? I want to move that onto the particles, soon

};

struct CrystalRenderConfig {
    // needs to include:
        // number of particles to display
        // light direction... potentially point light locations
        // theta, phi orbit camera controls
        // transform for the voxel model
        // debug flags
            // show bounding volume (outlines positive bounding volume hits)
            // show number of steps into the volume - heatmaps
            // show glyph bounds
        // include labels or not
}
//=================================================================================================
// global constants
constexpr int gifDelay = 4;
constexpr int imageWidth = 1280;
constexpr int imageHeight = 720;
constexpr int numPixels = imageWidth * imageHeight;
constexpr int NUM_THREADS = 72;                     // threads of execution
constexpr int NUM_PARTICLES = 1'000;                // size of pool of particles for the job system
constexpr int pad = 1000;                           // some extra particles as a safety buffer
constexpr uintmax_t maxParticles = 35'000'000 + pad;// size of pool of preallocated particle storage locations
constexpr int GridCellMaxParticles = 128;           // this size might make sense to play with eventually
//=================================================================================================
struct GridCell {
    vector< shared_ptr< mat4 > > particles;         // backing storage
    shared_mutex mutex;                             // access mutex - shared_lock for read, unique_lock for write

    gridCell () { particles.reserve( GridCellMaxParticles ); }
    size_t GetCount ();
    void Add( const mat4 );

    // size_t GetCount () {
        // this is the non-exclusive read mutex
        // std::shared_lock lock( mutex );
        // return particles.size();
    // }

    // void Add( const mat4 &pTransform ) {
        // lock the exclusive mutex for writing
        // std::unique_lock lock( mutex );
        // if ( particles.size() < GridCellMaxParticles ) {
            // get a new pointer...
            // uintmax_t ptrIdx = getPointerIndex();

            // using the input argument, fill out the data
            // shared_ptr< mat4 > ptr = pointerPool[ ptrIdx ];
            // *ptr = pTransform;

            // pushing the pointer onto the list
            // particles.push_back( ptr );
        // }
    // }

    // shared_ptr< mat4 > Get( const int &idx ) {
        // locking the non-exclusive read mutex
        // std::shared_lock lock( mutex );
        // return particles[ idx ];
    // }

};
//=================================================================================================
// encapsulating simulation and rendering
class Crystal {
public:

    // managed logger, allowing for worker threads to submit timestamped messages
        // todo

    // threads managed by constructor/destructor
    thread monitorThread;
    thread workerThreads[ NUM_THREADS ];

    // used signal termination to the worker threads
    atomic< bool > threadKill = false;

    // job counter + screenshot trigger
    atomic_uintmax_t jobDispatch = 0;
    atomic_uintmax_t ssDispatch = imageWidth * imageHeight + 1; // prime it so that it will not indicate a screenshot at init

    // scratch buffer to render an image
    uint8_t imageBuffer[ numPixels * 4 ];

    // 7x7 fatfont extended ASCII font LUT, 1-channel
    uint8_t* fontLUT;

    // render config
    CrystalRenderConfig renderConfig;

    // render support functions
    shared_mutex imageMutex;                                    // unique_lock for external reading, shared_lock for writing
    bool ScreenshotIndicated( uint32_t &x, uint32_t &y );       // return true for computed idx < numPixels, return by ref computed x, y
    bool ParticleUpdateIndicated( uintmax_t &jobIdx );          // return true if there is a reasonable number of particles left, ref atomic job counter value
    void ClearImage ();                                         // clear the entire image to black
    void DrawPixel ( const uint32_t x, const uint32_t y );      // this will only touch that pixel's memory... still need sync for resources we read from

    // global sim resources (particle + pointer pools, ivec3->gridCell hashmap)    
    vector< vec4 > particleScratch { NUM_PARTICLES };           // state for floating particles (diffusion limit mechanism) 
    vector< shared_ptr< mat4 > > particlePool { maxParticles }; // preallocated memory buffer for particles, allows playback
    atomic_uintmax_t particlePoolAllocator { 0u };              // bump allocator for above, enforces order
    HashMap< ivec3, shared_ptr< gridCell > > anchoredParticles; // concurrent hashmap of grid cell management pointers

    // sim support functions
    void AnchorParticle ();
    void RespawnParticle ( const int i );
    void UpdateParticle ( const int i );

    // simulation config
    CrystalSimConfig simConfig;

    // monitor thread function
    void MonitorThreadFunction ();

    // worker thread function
    void WorkerThreadFunction ( const int id );
    
    Crystal () { // should take a YAML config or default to a randomly generated one
        // fill out the config struct based on the YAML, or generate random + YAML

    // SPAWNING THREADS
        // a monitor thread which maintains a set of data for the master to access
            // and also will control things like screenshots... I want that to be
            // triggered by an atomic int "render" dispatcher being reset to 0 by
            // this thread, and the worker threads each increment it to get a pixel
            // to work on... if it is greater than the number of pixels in the image
            // and the simulation has not completed, return to particle work

        // a set of threads which function as the worker threads for the job system


    // FONT LUT
        // it is somewhat redundant to parse this every time... but whatever, it's
            // just a small cost in the constructor, because it's not a large image

        int x, y, n; // last arg set to 1 means give me 1 channel data out 
        fontLUT = stbi_load( "fatFont.png", &x, &y, &n, 1 );
    }

    ~Crystal () {

        // need to signal termination to all the threads
        threadKill = true;
        sleep_for( 69ms );

        // join the monitor thread and all the worker threads

    }
};
//=================================================================================================
// screenshot utilities
//=================================================================================================
void Crystal::clearImage () {
    shared_lock( imageMutex );
    for ( auto& c : imageBuffer ) {
        c = 0;
    }
}
//=================================================================================================
// dependent on the current renderstate ( this is called by the worker threads )
void Crystal::DrawPixel ( const uint32_t x, const uint32_t y ) {
    constexpr float ratio = ( 21.0f / 9.0f );
    const vec2 uv = vec2( x + 0.5f, y + 0.5f ) / vec2( imageWidth, imageHeight );

    vec3 color = vec3( 0.0f );
    shared_lock( imageMutex );

    // width ( x ) is from 0 to 1... height ( y ) is based on the 21:9/~2.35:1 ratio
    const vec2 lowThresh = vec2( 0, 0.5f / ratio );
    const vec2 highThresh = vec2( 1, 1.0f - 0.5f / ratio );

    if ( glm::all( glm::lessThan( uv, highThresh ) ) &&
        glm::all( glm::greaterThan( uv, lowThresh ) ) ) {
        // we are in the area where we need to do additional work to render

        // ray setup

        // bounds intersection

        // delta track raymarch

            // shadow ray trace
        
    } else {
        // we can do string evaluation here, directly from the config struct...
            // this is actually a nice place to do it, because it will give the jobs that do work on
            // the black bars will have some work to do, evaluating against the list of characters

        // we should setup a list of glyphs and other features, and test against them in an efficient
            // way, here, to see if we need to draw... debug flag only evaluates glyph bounds 
    }
}
//=================================================================================================
// particle support functions
//=================================================================================================
bool Crystal::ScreenshotIndicated ( uint32_t &x, uint32_t &y ) {
    // no need to branch, we will do it one level up
    uintmax_t idx = ssDispatch.fetch_add( 1 );
    x = idx % imageWidth;
    y = idx / imageHeight;
    return ( idx < numPixels );
}
//=================================================================================================
bool Crystal::ParticleUpdateIndicated ( uintmax_t &idx ) {
    idx = jobDispatch.fetch_add( 1 );
    // this makes a weaker guarantee than the screenshot atomic...
    return ( particlePoolAllocator < ( maxParticles - pad ) );
}
//=================================================================================================
void Crystal::AnchorParticle () {
    // add the particle to the hashmap

}
//=================================================================================================
void Crystal::RespawnParticle ( const int i ) {
    // respawn the particle
    
}
//=================================================================================================
void Crystal::UpdateParticle ( const int i ) {
    // do an update on the particle
        
}
//=================================================================================================
// this is the master thread over the worker threads on the crystal object
void Crystal::MonitorThreadFunction () {
    // enter a loop
    while ( true ) {
        // how do we quit? something indicated from the master
            // we will need a number of atomic signals... screenshot(atomic) + config, quit(atomic),
            // reset(atomic)

        // triggering a screenshot is like, 

        // what do we monitor?
            // some kind of flag we watch to indicate master wants a screenshot
                // this should include render config so that animations etc can be orchestrated there
            // when you want to do a screenshot, if you need to prep data, do it here before cutting
                // over to the worker threads to do the work
            // some kind of percentage completion, so that we can report that at the higher level

        // wait for the next iteration
        sleep_for( 100ms );
    }
}
//=================================================================================================
// worker thread, doing the particle update
void Crystal::WorkerThreadFunction ( int id ) {
    // enter a loop...
    while ( !threadKill ) {
        /*
        If we hit the screenshot counter and get a value less than the number of pixels, we need
            to do work to render an image, instead of the particle update...

        We also need to prevent running out of pointers in the pool... so check the allocator
            if we are "close" to the maximum, we need to enter a waiting state...

        I'd like to have this be a recoverable state, so if you want to shuffle the pointers and
            reset the allocator counter at a lower number... you could resume the sim with some
            subset of the original points... not sure. I'd like to also include interactions with
            other hashmaps, essentially seeding from other crystals without copying all the points
            to this crystal's hashmap

        So basically:

        if ( screenshot indicated ) {

            do screenshot work

        } else if ( particle update indicated ) {

            do particle work

        } else { // some other way to force this?

            waiting 1ms

        }
                
        */
    }
}
//=================================================================================================