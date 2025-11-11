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
#include <shared_mutex>
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
using std::atomic;
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
using glm::clamp;

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
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((0.192919f+t*(1.618437f+t*(-39.426098f+t*(737.420549f+t*(-6489.216487f+t*(28921.755478f+t*(-72384.553891f+t*(107076.097978f+t*(-93276.212113f+t*(44337.286143f+t*-8884.508085f)))))))))),
                      (0.101988f+t*(1.859131f+t*(7.108520f+t*(-20.179546f+t*11.147684f)))),
                      (0.253316f+t*(4.858570f+t*(55.191710f+t*(-803.379980f+t*(4477.461997f+t*(-14496.039745f+t*(28438.311669f+t*(-32796.884355f+t*(20328.068712f+t*-5210.826342f)))))))))), 0.0f, 1.0f);
}

vec3 bone(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((-0.011603f+t*(1.066867f+t*(-0.673604f+t*0.623930f))),
                      (0.018446f+t*(0.524946f+t*(1.185768f+t*-0.732596f))),
                      (0.004421f+t*(1.254048f+t*-0.275214f))), 0.0f, 1.0f);
}

vec3 afmhot(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((-0.000000f+t*2.000000f),
                      (-0.500000f+t*2.000000f),
                      (-1.000000f+t*2.000000f)), 0.0f, 1.0f);
}

vec3 gist_heat(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((0.000000f+t*1.500000f),
                      (-1.000000f+t*2.000000f),
                      (-3.000000f+t*4.000000f)), 0.0f, 1.0f);
}

vec3 plasma(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((0.057526f+t*(2.058166f+t*-1.141244f)),
                      (-0.183275f+t*(0.668964f+t*0.479353f)),
                      (0.525210f+t*(1.351117f+t*(-4.013494f+t*2.284066f)))), 0.0f, 1.0f);
}

vec3 inferno(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((-0.015449f+t*(0.816640f+t*(3.399179f+t*(-4.796465f+t*1.530683f)))),
                      (0.000619f+t*(0.450682f+t*(-1.556978f+t*(3.904984f+t*-1.764423f)))),
                      (0.019123f+t*(0.792737f+t*(29.365333f+t*(-210.608893f+t*(622.120191f+t*(-942.393021f+t*(711.115854f+t*-209.780428f)))))))), 0.0f, 1.0f);
}

vec3 magma(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return clamp(vec3((-0.023114f+t*(0.883412f+t*(2.280390f+t*-2.164009f))),
                      (-0.000931f+t*(0.700294f+t*(-3.639731f+t*(14.399222f+t*(-28.183967f+t*(29.245012f+t*-11.549071f)))))),
                      (0.011971f+t*(1.223232f+t*(17.782054f+t*(-111.294284f+t*(282.340184f+t*(-384.394777f+t*(275.310307f+t*-80.251736f)))))))), 0.0f, 1.0f);
}
//=================================================================================================
// random number generation utilities
#include "random.h"
// RNG objects for uniform and normal distributions
inline thread_local rng uniformRNG( 0.0f, 1.0f );
inline thread_local rngN normalRNG( 0.0f, 0.1f );
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

    // chance to bond
    // what else?

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
};
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
// I want to eventually have each particle keep color, too... large pool + randomized colors would be very cool
    vector< shared_ptr< mat4 > > particles;         // backing storage
    shared_mutex mutex;                             // access mutex - shared_lock for read, unique_lock for write

    GridCell () { particles.reserve( GridCellMaxParticles ); }

    // size_t GetCount () {
        // this is the non-exclusive read mutex
        // std::shared_lock lock( mutex );
        // return particles.size();
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
    atomic_uintmax_t ssDispatch = numPixels + 1;                // prime it so that it will not indicate a screenshot at init

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
    void DrawPixel ( uint32_t x, uint32_t y );                  // this will only touch that pixel's memory... still need sync for resources we read from
    bool GlyphRef ( const uint8_t &c, ivec2 offset ) const;     // reference the font LUT
    void SaveCurrentImage ( const string &filename );           // save out the current framebuffer state
    void StampString ( const string &s, ivec2 location, ivec2 scale ); // write a string onto the image
    void StampChar ( const uint8_t &c, ivec2 location, ivec2 scale ); // called by StampString

    // global sim resources (particle + pointer pools, ivec3->gridCell hashmap)    
    vector< vec4 > particleScratch{ NUM_PARTICLES };            // state for floating particles (diffusion limit mechanism)
    vector< shared_ptr< mat4 > > particlePool{ maxParticles };  // preallocated memory buffer for particles, allows playback
    atomic_uintmax_t particlePoolAllocator { 0u };              // bump allocator for above, enforces order
    HashMap< ivec3, shared_ptr< GridCell > > anchoredParticles; // concurrent hashmap of grid cell management pointers

    // tracking sim parameters
    ivec3 minExtents = ivec3( 0 );
    ivec3 maxExtents = ivec3( 0 );

    // sim support functions
    void AnchorParticle ( int i, const mat4 &pTransform );
    void RespawnParticle ( int i );
    void UpdateParticle ( int i );

    // simulation config
    CrystalSimConfig simConfig;

    // monitor thread function
    void MonitorThreadFunction ();

    // worker thread function
    void WorkerThreadFunction ( int id );
    
    Crystal () { // should take a YAML config or default to a randomly generated one

    // SIM INIT
        // fill out the config struct based on the YAML, or generate random + YAML

        // respawn all the particles in the pool

        // add initial seed particles to the hashmap

        // create the importance sampling structure around 

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

        ClearImage();
        const string s = string( "TEST 123 TEST" );

        StampString( s, ivec2( 100, 200 ), ivec2( 1 ) );
        StampString( s, ivec2( 100, 250 ), ivec2( 1, 2 ) );
        StampString( s, ivec2( 100, 300 ), ivec2( 2, 1 ) );

        SaveCurrentImage( "test.png" );
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
inline void Crystal::ClearImage () {
    shared_lock lock( imageMutex );
    for ( int i = 0; i < 4 * numPixels; i += 4 ) {
        imageBuffer[ i ] = imageBuffer[ i + 1 ] = imageBuffer[ i + 2 ] = 0;
        imageBuffer[ i + 3 ] = 255;
    }
}
//=================================================================================================
// dependent on the current renderstate ( this is called by the worker threads )
inline void Crystal::DrawPixel ( const uint32_t x, const uint32_t y ) {
    constexpr float ratio = ( 21.0f / 9.0f );
    const vec2 uv = vec2( x + 0.5f, y + 0.5f ) / vec2( imageWidth, imageHeight );

    // width ( x ) is from 0 to 1... height ( y ) is based on the 21:9/~2.35:1 "cinematic" ratio
    const vec2 lowThresh = vec2( 0, 0.5f / ratio );
    const vec2 highThresh = vec2( 1, 1.0f - 0.5f / ratio );

    vec3 color = vec3( 0.0f );
    shared_lock lock( imageMutex );
    if ( glm::all( glm::lessThan( uv, highThresh ) ) &&
        glm::all( glm::greaterThan( uv, lowThresh ) ) ) {
        // we are in the area where we need to do additional work to render

        // ray setup

        // bounds intersection

        // delta track raymarch

            // shadow ray trace
        
    } // else you are in the black bars area
        // I want to do the labels single threaded, not much sense doing it outside

    // write the color to the image
    const uint32_t baseIdx = 4 * ( x + imageWidth * y );
    imageBuffer[ baseIdx + 0 ] = color.r;
    imageBuffer[ baseIdx + 1 ] = color.g;
    imageBuffer[ baseIdx + 2 ] = color.b;
    imageBuffer[ baseIdx + 3 ] = 255;
}
//=================================================================================================
inline bool Crystal::GlyphRef ( const uint8_t &c, const ivec2 offset ) const {
    const ivec2 baseGlyphLoc = 7 * ivec2( ( c % 16 ), ( c / 16 ) );
    const ivec2 samplePoint = baseGlyphLoc + offset;
    return fontLUT[ samplePoint.x + samplePoint.y * 7 * 16 ];
}
//=================================================================================================
inline void Crystal::StampString ( const string &s, ivec2 location, const ivec2 scale ) {
    for ( const uint8_t c : s ) {
        StampChar( c, location, scale );
        location.x += scale.x * 8; // 7px + 1px pad
    }
}
//=================================================================================================
inline void Crystal::StampChar ( const uint8_t &c, ivec2 location, const ivec2 scale ) {
    for ( int y = 0; y < 7 * scale.y; y++ ) {
        for ( int x = 0; x < 7 * scale.x; x++ ) {
            const ivec2 writeLoc = location + ivec2( x, y );
            const ivec2 px = ivec2( x / scale.x, y / scale.y );

            // check px against the font LUT
            if ( GlyphRef( c, px ) ) {
                uint32_t idx = 4 * ( writeLoc.x + imageWidth * writeLoc.y );
                imageBuffer[ idx + 0 ] = 255;
                imageBuffer[ idx + 1 ] = 255;
                imageBuffer[ idx + 2 ] = 255;
                imageBuffer[ idx + 3 ] = 255;
            }
        }
    }
}
//=================================================================================================
inline void Crystal::SaveCurrentImage ( const string &filename ) {
    stbi_write_png( filename.c_str(), imageWidth, imageHeight, 4, &imageBuffer[ 0 ], 4 * imageWidth );
}
//=================================================================================================
// particle support functions
//=================================================================================================
inline bool Crystal::ScreenshotIndicated ( uint32_t &x, uint32_t &y ) {
    // no need to branch, we will do it one level up
    uintmax_t idx = ssDispatch.fetch_add( 1 );
    x = idx % imageWidth;
    y = idx / imageHeight;
    return ( idx < numPixels );
}
//=================================================================================================
inline bool Crystal::ParticleUpdateIndicated ( uintmax_t &jobIdx ) {
    jobIdx = jobDispatch.fetch_add( 1 );
    // this makes a weaker guarantee than the screenshot atomic...
    return ( particlePoolAllocator < ( maxParticles - pad ) );
}
//=================================================================================================
// probably add the allocator function here

//=================================================================================================
// add the particle to the hashmap
inline void Crystal::AnchorParticle ( const int i, const mat4 &pTransform ) {
    const ivec3 iP = ivec3( particleScratch[ i ].xyz() );
    shared_ptr< GridCell > gcp;
    if ( anchoredParticles.find( iP, gcp ) ) {
        // a GridCell already exists...
            // we get the pointer to it in gcp
    } else {
        // we haven't added to this GridCell yet
        gcp = make_shared< GridCell >();
    }

    { // lock the GridCell write mutex and add this point, if we have space
        unique_lock lock( gcp->mutex );
        if ( gcp->particles.size() < GridCellMaxParticles ) {
            // increment the bump allocator
            const shared_ptr< mat4 > ptr = particlePool[ particlePoolAllocator.fetch_add( 1 ) ];
            // populate the value and add it to the GridCell
            *ptr = pTransform;
            gcp->particles.push_back( ptr );
        }
    }

    // update statistics
    minExtents = glm::min( iP, minExtents );
    maxExtents = glm::max( iP, maxExtents );
}
//=================================================================================================
// respawn the particle
inline void Crystal::RespawnParticle ( const int i ) {
    // simple importance sampling scheme

}
//=================================================================================================
// do an update on the particle
inline void Crystal::UpdateParticle ( const int i ) {

}
//=================================================================================================
// this is the master thread over the worker threads on the crystal object
inline void Crystal::MonitorThreadFunction () {
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
inline void Crystal::WorkerThreadFunction ( int id ) {
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