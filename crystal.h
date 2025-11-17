//=================================================================================================
// basic IO
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
using std::cout;
using std::endl;
using std::string;
using std::to_string;
//=================================================================================================
const inline string frontPad ( int width, string s ) {
    return string( width - s.length(), ' ' ) + s;
}
//=================================================================================================
// Terminal UI output
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>

#include <ftxui/dom/table.hpp>
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
using std::make_unique;
using std::unique_ptr;
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
using glm::ivec4;
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
			std::size_t h1 = std::hash< int >{}( s.x + 512 );
			std::size_t h2 = std::hash< int >{}( s.y + 512 );
			std::size_t h3 = std::hash< int >{}( s.z + 512 );
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
// YAML lib
#include "yaml-cpp-0.8.0/include/yaml-cpp/yaml.h"
namespace YAML {
    // encode/decode for vector types (add more as we go)
    template<>
    struct convert<vec3> {
        static Node encode( const vec3& rhs ) {
            Node node;
            node.push_back( rhs.x );
            node.push_back( rhs.y );
            node.push_back( rhs.z );
            return node;
        }

        static bool decode( const Node& node, vec3& rhs ) {
            if ( !node.IsSequence() || node.size() != 3 ) {
                return false;
            }

            rhs.x = node[ 0 ].as<float>();
            rhs.y = node[ 1 ].as<float>();
            rhs.z = node[ 2 ].as<float>();
            return true;
        }
    };
    template<>
    struct convert<ivec3> {
        static Node encode( const ivec3& rhs ) {
            Node node;
            node.push_back( rhs.x );
            node.push_back( rhs.y );
            node.push_back( rhs.z );
            return node;
        }

        static bool decode( const Node& node, ivec3& rhs ) {
            if ( !node.IsSequence() || node.size() != 3 ) {
                return false;
            }

            rhs.x = node[ 0 ].as<int>();
            rhs.y = node[ 1 ].as<int>();
            rhs.z = node[ 2 ].as<int>();
            return true;
        }
    };
}
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

    // integral stuff:
    // number of threads -> difficult-ish to change at runtime, skip it
    uint32_t numParticlesScratch; // size of particle scratch pool
    uint32_t numParticlesStorage; // size of particle storage pool

    // bonding offsets + template information or "random" and we list the bonding offsets
    // templating for the basic crystal types
    string bondingOffsetTemplate;
    float bondingOffsetTemplateValues[ 6 ]; // a,b,c,alpha,beta,gamma
    // do we need something to represent symmetry?
    vector< vec4 > bondingOffsets; // simulation runtime bonding offsets

    // scale factor for particle jitter
    float temperature;

    // search radius? longer term, maybe

    // max bonding distance + stochastic bonding
    float bondThreshold;
    float chanceToBond;

    // bonding strategy - random of the bonding offsets available or closest bonding offset
        // maybe this can be like "chance to randomize"
    float bondChanceToRandomize;

    // static flow, dynamic flow, "wind" terms
    float staticFlowAmount;
    vec3 staticFlowDirection;
    float dynamicFlowAmount;
    vec3 dynamicFlowInitialDirection;

    // defects
    float defectRate; // how often a defect happens
    float defectPositionJitter; // how big the position jitter is
    float defectRotationJitter; // how big the rotation jitter is

    // attrition health + oob margin
    float particleAttitionHealth;
    float particleOOBMargin;

    // information to create the spawn importance sampling structure
    float spawnProbabilities[ 7 ];
    uint8_t importanceStructure[ 1024 ];

    // informing the initial seeding
    ivec3 InitialSeedSpanMin;
    ivec3 InitialSeedSpanMax;
    uint32_t numInitialSeedParticles;

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

    bool pixelShuffle = true;
    vector< ivec2 > offsets;

    float outputScalar = 4.0f;

    ivec3 minExtentComputed;
    ivec3 maxExtentComputed;

    mat4 transform;
};
//=================================================================================================
// global constants
constexpr int gifDelay = 4;
constexpr int imageWidth = 1280;
constexpr int imageHeight = 720;
constexpr int numPixels = imageWidth * imageHeight;
constexpr int NUM_THREADS = 1;                     // threads of execution
constexpr int pad = 1000;                           // some extra particles as a safety buffer
constexpr int GridCellMaxParticles = 128;           // this size might make sense to play with eventually
//=================================================================================================
struct GridCell {
// I want to eventually have each particle keep color, too... large pool + randomized colors would be very cool
    vector< shared_ptr< mat4 > > particles;         // backing storage
    shared_mutex mutex;                             // access mutex - shared_lock for read, unique_lock for write

    GridCell () { particles.reserve( GridCellMaxParticles ); }

    size_t GetCount () {
        shared_lock lock( mutex ); // this is the non-exclusive read mutex
        return particles.size();
    }

    shared_ptr< mat4 > Get( const int &idx ) {
        shared_lock lock( mutex ); // locking the non-exclusive read mutex
        return particles[ idx ];
    }

};
//=================================================================================================
// encapsulating simulation and rendering
class Crystal {
public:

    // managed logger, allowing for worker threads to submit timestamped messages
    // todo

    // stuff for reporting state to the terminal UI thread
    string GetStateString();
    float GetPercentage();

    // some placeholder stuff, hooks for controls
    void Screenshot( string filename );
    void Save();
    void Shutdown();

    // threads managed by constructor/destructor
    thread monitorThread;
    thread workerThreads[ NUM_THREADS ];

    // used signal termination to the worker threads
    atomic< bool > threadKill = false;
    atomic< bool > pause = false;
    atomic< bool > renderPrep = false;
    atomic< float > renderPrepEstimatedCompletion = 0.0f;
    atomic< bool > imageSaving = false;

    // job counter + screenshot trigger
    atomic_uintmax_t jobDispatch = 0;
    atomic_uintmax_t ssDispatch = numPixels + 1;                // prime it so that it will not indicate a screenshot at init
    atomic_uintmax_t ssComplete = 0;
    mutex ssMutex;

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
    void SaveCurrentImage ( const string &filename ) const;     // save out the current framebuffer state
    void StampString ( const string &s, ivec2 location, ivec2 scale, ivec4 color ); // write a string onto the image
    void StampChar ( const uint8_t &c, ivec2 location, ivec2 scale, ivec4 color ); // called by StampString

    // global sim resources (particle + pointer pools, ivec3->gridCell hashmap)    
    vector< vec4 > particleScratch;                             // state for floating particles (diffusion limit mechanism)
    vector< shared_ptr< mat4 > > particleStorage;               // preallocated memory buffer for particles, allows playback
    atomic_uintmax_t particleStorageAllocator;                  // bump allocator for above, enforces order
    HashMap< ivec3, shared_ptr< GridCell > > anchoredParticles; // concurrent hashmap of grid cell management pointers
    HashMap< ivec3, vec4 > voxelModel;                          // lighter weight version of the grid for rendering

    // tracking sim parameters
    ivec3 minExtents = ivec3( 1000 );
    ivec3 maxExtents = ivec3( -1000 );

    // sim support functions
    void AnchorParticle ( ivec3 iP, const mat4 &pTransform );
    void RespawnParticle ( int i );
    void UpdateParticle ( int i );

    // simulation config
    CrystalSimConfig simConfig;
    void LoadSpecifiedConfig ( const string& path );
    void GenerateRandomConfig ();
    void GenerateOffsets ( const string& templateSelect );

    // monitor thread function
    void MonitorThreadFunction ();
    atomic< float > percentageCompletion = 0.0f;

    // worker thread function
    void WorkerThreadFunction ( int id );

    Crystal ( string yamlPath );
    ~Crystal () {
        // need to signal termination to all the threads
        threadKill = true;
        sleep_for( 69ms );

        // join the monitor thread and all the worker threads
        monitorThread.join();
        for ( auto& t : workerThreads )
            t.join();

        Shutdown();
    }
};
//=================================================================================================
// for monitoring state
inline string Crystal::GetStateString () {
    // options are "WORKING", "RENDERING", "FINISHED"
    if ( ssComplete < numPixels && ssDispatch < numPixels )
        return string( "RENDERING" );
    else if ( renderPrep )
        return string( "RENDER PREP" );
    else if ( imageSaving )
        return string( "IMAGE SAVING" );
    else if ( particleStorageAllocator < ( simConfig.numParticlesStorage - pad ) )
        return string( "WORKING" );
    else
        return string( "FINISHED" );
}

inline float Crystal::GetPercentage () {
    string ss = GetStateString(); 
    if ( ss == "WORKING" )
        return float( particleStorageAllocator ) / float( simConfig.numParticlesStorage );
    else if ( ss == "RENDERING" )
        return float( ssComplete ) / float( numPixels );
    else if ( ss == "RENDER PREP" )
        return renderPrepEstimatedCompletion;
    else // "FINISHED"
        return 1.0f;
}
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

    const vec3 minExtentsIn = renderConfig.minExtentComputed;
    const vec3 maxExtentsIn = renderConfig.maxExtentComputed;

    vec3 color = vec3( 0.0f );
    int hits = 0;
    constexpr int samples = 128;
    
    shared_lock lock( imageMutex );
    if ( glm::all( glm::lessThan( uv, highThresh ) ) &&
        glm::all( glm::greaterThan( uv, lowThresh ) ) ) {
        // we are in the area of the image where we need to do additional work to render

        // ray setup
        vec2 uvAdjust = ( uv - vec2( 0.5f, 0.5f ) ) * vec2( 1.0f, float( imageHeight ) / float( imageWidth ) );
        int maxDistance = int( std::ceil( glm::distance( vec3( minExtentsIn ), vec3( maxExtentsIn ) ) ) );
        mat4 inverseTransform = glm::inverse( renderConfig.transform );
        vec3 rO = inverseTransform * vec4( vec3( std::max( 300.0f, float( maxDistance * 2 ) ) * uvAdjust, -200 ), 1.0f );
        vec3 rD = inverseTransform * vec4( 0.0f, 0.0f, 1.0f, 0.0f );

        // bounds intersection
        float tMin, tMax;
        if ( IntersectAABB( rO, rD, minExtentsIn, maxExtentsIn, tMin, tMax ) ) {

            // delta track raymarch
            const vec3 p0 = rO + max( 0.0f, tMin ) * rD;
            for ( int s = 0; s < samples; s++ ) {
                // vec3 p = p0 + vec3( normalRNG(), normalRNG(), 0.0f );
                vec3 p = p0;
                for ( int i = 0; i < maxDistance; i++ ) {
                    float t = -log( uniformRNG() );
                    p += t * rD;

                    if ( glm::any( glm::lessThanEqual( p, vec3( minExtentsIn ) ) ) ||
                        glm::any( glm::greaterThanEqual( p, vec3( maxExtentsIn ) ) ) ) {
                        // oob
                        break;
                    }

                    vec4 temp;
                    if ( voxelModel.find( ivec3( p ), temp ) ) {
                        if ( ( temp.a ) > uniformRNG() ) { // this is the hit condition...
                            // do we hit something, going up?
                            vec3 pShadow = p;
                            vec3 shadowTerm = vec3( 1.0f );

                            const vec3 dir1 = normalize( p - vec3( mix( minExtents.xy(), maxExtents.xy(), vec2( uniformRNG() ) ), maxExtentsIn.z + 5.0f ) );
                            // const vec3 dir2 = vY;
                            // const vec3 dir3 = vZ;

                            // shadow ray trace(s)
                            for ( int j = 0; j < 200; j++ ) {
                            // light direction needs to go on renderconfig
                                const vec3 lightDirection = dir1;
                                pShadow += lightDirection * float( -log( uniformRNG() ) );
                                vec4 tempShadow;
                                if ( voxelModel.find( ivec3( pShadow ), tempShadow ) ) {
                                    if ( ( temp.a ) > uniformRNG() ) {
                                        shadowTerm.r = 0.1f;
                                        break;
                                    }
                                }
                            }

                            /*
                            pShadow = p;
                            for ( int j = 0; j < 20; j++ ) {
                            // light direction needs to go on renderconfig
                                const vec3 lightDirection = dir2;
                                pShadow += lightDirection * float( -log( uniformRNG() ) );
                                vec4 tempShadow;
                                if ( voxelModel.find( ivec3( pShadow ), tempShadow ) ) {
                                    if ( ( temp.a ) > uniformRNG() ) {
                                        shadowTerm.g = 0.1f;
                                    }
                                }
                            }

                            pShadow = p;
                            for ( int j = 0; j < 20; j++ ) {
                            // light direction needs to go on renderconfig
                                const vec3 lightDirection = dir3;
                                pShadow += lightDirection * float( -log( uniformRNG() ) );
                                vec4 tempShadow;
                                if ( voxelModel.find( ivec3( pShadow ), tempShadow ) ) {
                                    if ( ( temp.a ) > uniformRNG() ) {
                                        shadowTerm.b = 0.1f;
                                    }
                                }
                            }
                            */

                            //  color needs to go on renderconfig
                            color += temp.rgb() * vec3( shadowTerm.r );
                            // color += turbo( float( i ) / float( maxDistance ) );
                            hits++;
                            // break;
                            continue;
                        }
                    }
                }
            }
        }
    } // else you are in the black bars area
        // I want to do the labels single threaded, not much sense making it
        // more complicated trying to evaluate a list of glyphs here

    color /= float( 1.2f * hits );

    // write the color to the image... this might move into the condition because we already overwrote with black opaque
    const uint32_t baseIdx = 4 * ( x + imageWidth * y );
    imageBuffer[ baseIdx + 0 ] = std::clamp( 255.0f * color.r, 0.0f, 255.0f );
    imageBuffer[ baseIdx + 1 ] = std::clamp( 255.0f * color.g, 0.0f, 255.0f );
    imageBuffer[ baseIdx + 2 ] = std::clamp( 255.0f * color.b, 0.0f, 255.0f );
    imageBuffer[ baseIdx + 3 ] = 255;
}
//=================================================================================================
inline bool Crystal::GlyphRef ( const uint8_t &c, const ivec2 offset ) const {
    const ivec2 baseGlyphLoc = 7 * ivec2( ( c % 16 ), ( c / 16 ) );
    const ivec2 samplePoint = baseGlyphLoc + offset;
    return fontLUT[ samplePoint.x + samplePoint.y * 7 * 16 ];
}
//=================================================================================================
inline void Crystal::StampString ( const string &s, ivec2 location, const ivec2 scale, const ivec4 color ) {
    for ( const uint8_t c : s ) {
        StampChar( c, location, scale, glm::clamp( color, ivec4( 0 ), ivec4( 255 ) ) );
        location.x += scale.x * 8; // 7px + 1px pad
    }
}
//=================================================================================================
inline void Crystal::StampChar ( const uint8_t &c, ivec2 location, const ivec2 scale, const ivec4 color ) {
    for ( int y = 0; y < 7 * scale.y; y++ ) {
        for ( int x = 0; x < 7 * scale.x; x++ ) {
            const ivec2 writeLoc = location + ivec2( x, y );
            const ivec2 px = ivec2( x / scale.x, y / scale.y );

            // check px against the font LUT
            if ( GlyphRef( c, px ) ) {
                uint32_t idx = 4 * ( writeLoc.x + imageWidth * writeLoc.y );
                // if ( color.a != 255 ) {
                    // we are doing serial computation, no data races, let's do the alpha blending...
                // }
                imageBuffer[ idx + 0 ] = color.r;
                imageBuffer[ idx + 1 ] = color.g;
                imageBuffer[ idx + 2 ] = color.b;
                imageBuffer[ idx + 3 ] = color.a;
            }
        }
    }
}
//=================================================================================================
inline void Crystal::SaveCurrentImage ( const string &filename ) const {
    // unique_lock lock( imageMutex );
    stbi_write_png( filename.c_str(), imageWidth, imageHeight, 4, &imageBuffer[ 0 ], 4 * imageWidth );
}
//=================================================================================================
// External controls/public interface
//=================================================================================================
Crystal::Crystal ( const string yamlPath = "RANDOM" ) {
// SIM INIT
    // fill out the config struct based on the YAML, or generate random + YAML
        // constructor should take a YAML config or default to a randomly generated one
    if ( yamlPath == "RANDOM" ) {
        GenerateRandomConfig();
    } else {
        LoadSpecifiedConfig( yamlPath );
    }

    { // memory allocations
        particleScratch.resize( simConfig.numParticlesScratch );
        particleStorage.resize( simConfig.numParticlesStorage );
        particleStorageAllocator = 0;

        for ( auto& ptr : particleStorage ) {
            // shared_ptr initialized as nullptr, interesting
            ptr = make_shared< mat4 >();
        }
    }

    { // create the importance sampling structure around spawning particles on faces/in the volume
        float sum = 0.0f;
        for ( const auto& s : simConfig.spawnProbabilities ) {
            sum += s;
        }

        for ( auto& v : simConfig.importanceStructure ) {
            // constructing a set of integer indices for the faces and uniform volume spawn to
                // uniformly pick at runtime to preferentially select based on probabilities
            const float value = uniformRNG();
            float sum2 = 0.0f;
            for ( int i = 0; i < 7; i++ ) {
                sum2 += simConfig.spawnProbabilities[ i ] / sum;
                if ( value < sum2 ) {
                    v = i; // this index is selected by the threshold check
                    break;
                }
            }
        }
    }

    { // respawn all the particles in the pool
        for ( int i = 0; i < simConfig.numParticlesScratch; i++ ) {
            RespawnParticle( i );
        }
    }

    // cout << "adding initial anchored particles" << endl;
    { // add initial seed particles to the hashmap
        for ( int i = 0; i < simConfig.numInitialSeedParticles; i++ ) {
            // cout << "Adding particle " << i << endl;
            // pick from the specified distribution
            const vec3 p = glm::mix( vec3( simConfig.InitialSeedSpanMin ),
                vec3( simConfig.InitialSeedSpanMax ),
                vec3( uniformRNG(), uniformRNG(), uniformRNG() ) );

            const vec3 axis = normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) );
            const mat4 transform = glm::rotate( glm::translate( identity, p ), 10000.0f * uniformRNG(), axis );

            AnchorParticle( ivec3( p ), transform );
        }
    }

// FONT LUT
    // it is somewhat redundant to parse this every time... but whatever, it's
        // just a small cost in the constructor, because it's not a large image

    int x, y, n; // last arg set to 1 means give me 1 channel data out
    fontLUT = stbi_load( "fatFont.png", &x, &y, &n, 1 );

// SPAWNING THREADS
    // a monitor thread which maintains a set of data for the master to access
        // and also will control things like screenshots... I want that to be
        // triggered by an atomic int "render" dispatcher being reset to 0 by
        // this thread, and the worker threads each increment it to get a pixel
        // to work on... if it is greater than the number of pixels in the image
        // and the simulation has not completed, return to particle work

    // a set of threads which function as the worker threads for the job system

    monitorThread = thread( [ & ] () { MonitorThreadFunction(); } );
    for ( int i = 0; i < NUM_THREADS; i++ ) {
        workerThreads[ i ] = thread( [ & ] () { WorkerThreadFunction( i ); } );
    }
}

void Crystal::Screenshot ( string filename = "timestamp" ) {
    // spawn a thread to prepare the data, tbd
        // should do any prep work, clear the image, and then signal

    renderPrep = true;
    renderPrepEstimatedCompletion = 0.0f;

    if ( filename == string( "timestamp" ) ) {
        auto now = std::chrono::system_clock::now();
        auto inTime_t = std::chrono::system_clock::to_time_t( now );
        std::stringstream ssA;
        ssA << std::put_time( std::localtime( &inTime_t ), "Crystal-Screenshot-%Y-%m-%d at %H-%M-%S.png" );
        filename = ssA.str().c_str();
    }

    std::jthread t( [ &, filename ] () {
        lock_guard< mutex > lock( ssMutex ); // don't want to try to be doing more than one of these at once...

        // clear to black
        ClearImage();

        renderPrepEstimatedCompletion = 0.1f;

        // do any prep work for the image
        // calculate a transform
        renderConfig.transform = glm::rotate( identity, normalRNG() * 10.0f, normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) ) );
        
        // precompute lightweight model
        voxelModel.clear();
        int maxCount = 0;
        
        renderConfig.minExtentComputed = ivec3( 1000 );
        renderConfig.maxExtentComputed = ivec3( -1000 );
        const int count = int( particleStorageAllocator );
        const ivec3 minExtentsCache = minExtents;
        const ivec3 maxExtentsCache = maxExtents;

        const vec3 color1 = vec3( 0.6f );
        const vec3 color2 = vec3( 0.1f );

        for ( int i = 0; i < count; i++ ) {
            shared_ptr< mat4 > ptr = particleStorage[ i ];
            vec4 p = *ptr * p0;
            ivec3 iP = ivec3( renderConfig.outputScalar * p.xyz() );

            vec4 temp;
            vec4 col = vec4( glm::mix( color1, color2, float( i ) / float( count ) ), 1.0f );
            if ( voxelModel.find( iP, temp ) ) {
                temp += col;
            } else {
                temp = col;
            }

            renderConfig.minExtentComputed = min( renderConfig.minExtentComputed, iP );
            renderConfig.maxExtentComputed = max( renderConfig.maxExtentComputed, iP );
            maxCount = max( int( temp.a ), maxCount );
            voxelModel.insert( iP, temp );

            if ( !( i % 10000 ) ) {
                renderPrepEstimatedCompletion = 0.1f + 0.5f * ( float( i ) / float( count ) );
            }
        }

        // normalizing...
        for ( int x = renderConfig.minExtentComputed.x; x <= renderConfig.maxExtentComputed.x; x++ ) {
            for ( int y = renderConfig.minExtentComputed.y; y <= renderConfig.maxExtentComputed.y; y++ ) {
                for ( int z = renderConfig.minExtentComputed.z; z <= renderConfig.maxExtentComputed.z; z++ ) {
                    vec4 temp;
                    const ivec3 p = ivec3( x, y, z );
                    if ( voxelModel.find( p, temp ) ) {
                        temp.x = temp.x / temp.w; // averaged color
                        temp.y = temp.y / temp.w;
                        temp.z = temp.z / temp.w;
                        temp.w = ( temp.w / maxCount ); // normalized density
                        voxelModel.insert( p, temp );
                    }
                }
            }
            renderPrepEstimatedCompletion = 0.6f + 0.4f * ( 1.0f - float( x + renderConfig.minExtentComputed.x ) / ( renderConfig.minExtentComputed.x - renderConfig.maxExtentComputed.x ) );
        }

        renderPrep = false;
        renderPrepEstimatedCompletion = 1.0f;

        // job system cuts over to render work
        ssComplete = 0;
        ssDispatch = 0;

        // wait for it
        while ( ssComplete < numPixels ) {
            sleep_for( 1ms );
        }


        StampString( string( "Count: " ) + to_string( count ), ivec2( 100, 500 ), ivec2( 1 ), ivec4( 255, 189, 32, 255 ) );
        StampString( string( "Extents: " ), ivec2( 100, 510 ), ivec2( 1 ), ivec4( 255, 189, 32, 255 ) );
        StampString( string( " x: " ) + frontPad( 5, to_string( minExtentsCache.x ) ) + " " + frontPad( 5, to_string( maxExtentsCache.x ) ), ivec2( 100, 520 ), ivec2( 1 ), ivec4( 255, 189, 32, 255 ) );
        StampString( string( " y: " ) + frontPad( 5, to_string( minExtentsCache.y ) ) + " " + frontPad( 5, to_string( maxExtentsCache.y ) ), ivec2( 100, 530 ), ivec2( 1 ), ivec4( 255, 189, 32, 255 ) );
        StampString( string( " z: " ) + frontPad( 5, to_string( minExtentsCache.z ) ) + " " + frontPad( 5, to_string( maxExtentsCache.z ) ), ivec2( 100, 540 ), ivec2( 1 ), ivec4( 255, 189, 32, 255 ) );

        imageSaving = true;
        // save the image, now that it's done
        SaveCurrentImage( filename );
        imageSaving = false;
    } );

    // once the thread is spawned, we don't need to touch it...
    t.detach(); // jthreads automatically rejoin on destruction
}
//=================================================================================================
void Crystal::Shutdown () {
    // need to set threadkill?
        // save the crystal if we haven't saved in the last 10 seconds or so...
        // also do a screenshot if it's been running at least 10 seconds...
    threadKill = true;
}
//=================================================================================================
void Crystal::Save () {
    // save out the model

}
//=================================================================================================
// particle support functions
//=================================================================================================
inline bool Crystal::ScreenshotIndicated ( uint32_t &x, uint32_t &y ) {
    // no need to branch, we will do it one level up
    uintmax_t idx = ssDispatch.fetch_add( 1 );
    x = idx % imageWidth;
    y = idx / imageWidth;
    return ( idx < ( numPixels ) );
}
//=================================================================================================
inline bool Crystal::ParticleUpdateIndicated ( uintmax_t &jobIdx ) {
    jobIdx = jobDispatch.fetch_add( 1 );
    // this makes a weaker guarantee than the screenshot atomic...
    return ( particleStorageAllocator < ( simConfig.numParticlesStorage - pad ) );
}
//=================================================================================================
// probably add the allocator function here
//=================================================================================================
inline void Crystal::GenerateOffsets ( const string &templateSelect ) {

// config has:
    // string bondingOffsetTemplate;
    // float bondingOffsetTemplateValues[ 6 ];
    // vector< vec3 > bondingOffsets;

// for now, reusing the previous setup:
    const float xXx = 0.1618f + 0.3f * uniformRNG();
    const float yYy = 0.618f + 0.1f * normalRNG();
    const float zZz = 0.618f;

    const float pi = 3.1415926535f;
    const mat4 rX = glm::rotate( identity, 0.0f + uniformRNG(), normalize( vec3( 0.0f, 0.0f, 1.0f ) ) );
    const mat4 rY = glm::rotate( identity, uniformRNG() * pi / 2.0f, normalize( vec3( 0.0f, 0.0f, 1.0f ) ) );
    const mat4 rZ = glm::rotate( identity, pi / 5.0f, normalize( vec3( 1.0f, 0.0f, 0.0f ) ) );

    // note that 6 bonding points is in no way a constraint here
    simConfig.bondingOffsets = {
        // rZ * vec4( 0.0f, 0.0f, -zZz, 0.0f ),
        rZ * vec4( 0.0f, 0.0f, zZz, 0.0f ),
        // rY * vec4( 0.0f, -yYy, 0.0f, 0.0f ),
        rY * vec4( 0.0f, yYy, 0.0f, 0.0f ),
        // rX * vec4( -xXx, 0.0f, 0.0f, 0.0f ),
        rX * vec4( xXx, 0.0f, 0.0f, 0.0f ),
    };
}
//=================================================================================================
inline void Crystal::LoadSpecifiedConfig ( const string &path ) {
    // load a YAML string from a file
        // todo
}
//=================================================================================================
inline void Crystal::GenerateRandomConfig () {

    YAML::Emitter out;
    out << YAML::BeginMap;

    // not going to do a lot right now
    GenerateOffsets( "RANDOM" );
    // todo: add to the config YAML

    simConfig.numParticlesScratch = 1000;
    out << YAML::Key << "numParticlesScratch";
    out << YAML::Value << simConfig.numParticlesScratch;

    simConfig.numParticlesStorage = 50'000'000;
    out << YAML::Key << "numParticlesStorage";
    out << YAML::Value << simConfig.numParticlesStorage;

    simConfig.numInitialSeedParticles = std::max( uint32_t( 20.0f * uniformRNG() ), 1u );
    out << YAML::Key << "numInitialSeedParticles";
    out << YAML::Value << simConfig.numInitialSeedParticles;

    simConfig.InitialSeedSpanMin = ivec3( -100, -50, -10 );
    out << YAML::Key << "InitialSeedParticleSpanMinX";
    out << YAML::Value << simConfig.InitialSeedSpanMin.x;
    out << YAML::Key << "InitialSeedParticleSpanMinY";
    out << YAML::Value << simConfig.InitialSeedSpanMin.y;
    out << YAML::Key << "InitialSeedParticleSpanMinZ";
    out << YAML::Value << simConfig.InitialSeedSpanMin.z;

    simConfig.InitialSeedSpanMax = ivec3( 100, 50, 10 );
    out << YAML::Key << "InitialSeedParticleSpanMaxX";
    out << YAML::Value << simConfig.InitialSeedSpanMax.x;
    out << YAML::Key << "InitialSeedParticleSpanMaxY";
    out << YAML::Value << simConfig.InitialSeedSpanMax.y;
    out << YAML::Key << "InitialSeedParticleSpanMaxZ";
    out << YAML::Value << simConfig.InitialSeedSpanMax.z;

    simConfig.particleAttitionHealth = 69;
    out << YAML::Key << "particleAttitionHealth";
    out << YAML::Value << simConfig.particleAttitionHealth;

    simConfig.particleOOBMargin = 0;
    out << YAML::Key << "particleOOBMargin";
    out << YAML::Value << simConfig.particleOOBMargin;

    // initialize spawn probabilities as uniformly random
    for ( auto& sp : simConfig.spawnProbabilities ) {
        sp = 0.5f;
    }
    out << YAML::Key << "spawnProbabilityPositiveX";
    out << YAML::Value << simConfig.spawnProbabilities[ 0 ];
    out << YAML::Key << "spawnProbabilityNegativeX";
    out << YAML::Value << simConfig.spawnProbabilities[ 1 ];
    out << YAML::Key << "spawnProbabilityPositiveY";
    out << YAML::Value << simConfig.spawnProbabilities[ 2 ];
    out << YAML::Key << "spawnProbabilityNegativeY";
    out << YAML::Value << simConfig.spawnProbabilities[ 3 ];
    out << YAML::Key << "spawnProbabilityPositiveZ";
    out << YAML::Value << simConfig.spawnProbabilities[ 4 ];
    out << YAML::Key << "spawnProbabilityNegativeZ";
    out << YAML::Value << simConfig.spawnProbabilities[ 5 ];

    // uniform spawn seems to create the best actual "crystal" behavior, so we will bias towards that
    simConfig.spawnProbabilities[ 6 ] = 15.0f;
    out << YAML::Key << "spawnProbabilityUniformVolume";
    out << YAML::Value << simConfig.spawnProbabilities[ 6 ];

    simConfig.temperature = 10.0f + 10.0f * uniformRNG();
    out << YAML::Key << "temperature";
    out << YAML::Value << simConfig.temperature;

    simConfig.defectRate = 0.001f * uniformRNG();
    out << YAML::Key << "defectRate";
    out << YAML::Value << simConfig.defectRate;

    simConfig.defectPositionJitter = 10.0f;
    out << YAML::Key << "defectPositionJitter";
    out << YAML::Value << simConfig.defectPositionJitter;

    simConfig.defectRotationJitter = 3.0f;
    out << YAML::Key << "defectRotationJitter";
    out << YAML::Value << simConfig.defectRotationJitter;

    simConfig.bondChanceToRandomize = 0.0f;
    out << YAML::Key << "bondChanceToRandomize";
    out << YAML::Value << simConfig.bondChanceToRandomize;

    simConfig.bondThreshold = 0.75f;
    out << YAML::Key << "bondThreshold";
    out << YAML::Value << simConfig.bondThreshold;

    simConfig.chanceToBond = 1.0f;
    out << YAML::Key << "chanceToBond";
    out << YAML::Value << simConfig.chanceToBond;

    simConfig.staticFlowAmount = 0.0f;
    out << YAML::Key << "staticFlowAmount";
    out << YAML::Value << simConfig.staticFlowAmount;

    simConfig.staticFlowDirection = normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) );
    out << YAML::Key << "staticFlowDirectionX";
    out << YAML::Value << simConfig.staticFlowDirection.x;
    out << YAML::Key << "staticFlowDirectionY";
    out << YAML::Value << simConfig.staticFlowDirection.y;
    out << YAML::Key << "staticFlowDirectionZ";
    out << YAML::Value << simConfig.staticFlowDirection.z;

    simConfig.dynamicFlowAmount = 0.01f;
    out << YAML::Key << "dynamicFlowAmount";
    out << YAML::Value << simConfig.dynamicFlowAmount;

    simConfig.dynamicFlowInitialDirection = normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) );
    out << YAML::Key << "dynamicFlowInitialDirectionX";
    out << YAML::Value << simConfig.dynamicFlowInitialDirection.x;
    out << YAML::Key << "dynamicFlowInitialDirectionY";
    out << YAML::Value << simConfig.dynamicFlowInitialDirection.y;
    out << YAML::Key << "dynamicFlowInitialDirectionZ";
    out << YAML::Value << simConfig.dynamicFlowInitialDirection.z;

    out << YAML::EndMap;

    // cout << "Configured:" << endl;
    // cout << out.c_str() << endl;
}
//=================================================================================================
//=================================================================================================
// add the particle to the hashmap
inline void Crystal::AnchorParticle ( const ivec3 iP, const mat4 &pTransform ) {
    shared_ptr< GridCell > gcp;
    if ( anchoredParticles.find( iP, gcp ) ) {
        // a GridCell already exists...
            // we get the pointer to it in gcp
    } else {
        // we haven't added to this GridCell yet
        gcp = make_shared< GridCell >();
        anchoredParticles.insert( iP, gcp );
    }

    { // lock the GridCell write mutex and add this point, if we have space
        unique_lock lock( gcp->mutex );
        if ( gcp->particles.size() < GridCellMaxParticles ) {
            // increment the bump allocator
            const shared_ptr< mat4 > ptr = particleStorage[ particleStorageAllocator.fetch_add( 1 ) ];
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
    const uint8_t pick = simConfig.importanceStructure[ std::clamp( int( uniformRNG() * 1024 ), 0, 1023 ) ];

    const vec3 minE = vec3( minExtents );
    const vec3 maxE = vec3( maxExtents );
    const vec3 lerpValues = vec3( uniformRNG(), uniformRNG(), uniformRNG() );

    // uniformly distributed within the sim volume
    vec3 p = glm::mix( minE, maxE, lerpValues );

    /*
    switch ( pick ) {
    // flatten to one of the x faces
    case 0: p.x = minE.x; break;
    case 1: p.x = maxE.x; break;
    // flatten to one of the y faces
    case 2: p.y = minE.y; break;
    case 3: p.y = maxE.y; break;
    // flatten to one of the z faces
    case 4: p.z = minE.z; break;
    case 5: p.z = maxE.z; break;
    // uniform spawn - no change
    default: break;
    }
    */

    particleScratch[ i ] = vec4( p, simConfig.particleAttitionHealth );
}
//=================================================================================================
// do an update on the particle
inline void Crystal::UpdateParticle ( const int i ) {
    // AnchorParticle( ivec3( int( 100.0f * uniformRNG() ), int( 100.0f * uniformRNG() ), int( 100.0f * uniformRNG() ) ), mat4( 1.0f ) );
    int idx = i % simConfig.numParticlesScratch;
    vec4 &particle = std::ref( particleScratch[ idx ]  );

    // oob decrement + respawn logic
    if ( glm::any( glm::lessThanEqual( particle.xyz(), vec3( minExtents - ivec3( 10 ) ) ) ) ||
        glm::any( glm::greaterThanEqual( particle.xyz(), vec3( maxExtents + ivec3( 10 ) ) ) ) ) {
        // idea is that when you stray outside of the bounding volume, you suffer some attrition...
        particle.w--;
        if ( particle.w < 0.0f ) {
            // and eventually when you die, you respawn somewhere on the boundary
            RespawnParticle( idx );
        }
    }

    thread_local vec4 flowVector = vec4( simConfig.dynamicFlowInitialDirection, 0.0f );
    flowVector = glm::rotate( identity, normalRNG(), glm::normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) ) ) * flowVector;
    particle += vec4( simConfig.temperature * vec3( normalRNG(), normalRNG(), normalRNG() ) + uniformRNG() * flowVector.xyz() * simConfig.dynamicFlowAmount + uniformRNG() * simConfig.staticFlowDirection * simConfig.staticFlowAmount, 0.0f );

    // construct a list of nearby points...
    vector< mat4 > nearbyPoints;

    // looking at the local neighborhood...
    const auto& s = { -1, 0, 1 };
    for ( int x : s )
    for ( int y : s )
    for ( int z : s ) {
        // if ( x == 0 && y == 0 && z == 0 ) continue;
        const ivec3 loc = ivec3( particle.xyz() ) + ivec3( x, y, z );

        // "find()" will also return the pointer to the container contents
        if ( std::shared_ptr< GridCell > gcp; anchoredParticles.find( loc, gcp ) ) {
            // we have located a cell which contains particles...
            const size_t count = gcp->GetCount();

            for ( int j = 0; j < count; j++ ) {
                nearbyPoints.push_back( *gcp->Get( j ) );
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
        if ( closestPointDistance < simConfig.bondThreshold ) {
            // close enough... random hash... etc
            // figure out which of bonding sites you want to bond to... probably the closest one of them
            vec3 closestBondingPointOffset;
            float closestBondingPointDistance = 10000.0f;

            if ( uniformRNG() < simConfig.bondChanceToRandomize ) {
                // just pick one at random...
                closestBondingPointOffset = closestPointTransform * simConfig.bondingOffsets[ std::clamp( size_t( floor( uniformRNG() * simConfig.bondingOffsets.size() ) ), size_t( 0 ), simConfig.bondingOffsets.size() ) ];
            } else {
                for ( auto& bpo : simConfig.bondingOffsets ) {
                    vec4 transformedBondingPointOffset = closestPointTransform * bpo;
                    const float d = glm::distance( particle.xyz(), closestPointTransformed + transformedBondingPointOffset.xyz() );
                    if ( d < closestBondingPointDistance ) {
                        closestBondingPointDistance = d;
                        closestBondingPointOffset = transformedBondingPointOffset.xyz();
                    }
                }
            }

            if ( uniformRNG() < simConfig.defectRate ) {
                closestBondingPointOffset += simConfig.defectPositionJitter * vec3( normalRNG(), normalRNG(), normalRNG() );
                closestPointTransform = glm::translate( glm::rotate( glm::translate( closestPointTransform, -closestPointTransformed ), simConfig.defectRotationJitter * normalRNG(), glm::normalize( vec3( normalRNG(), normalRNG(), normalRNG() ) ) ), closestPointTransformed );
            }

            const vec4 TvX = closestPointTransform * vX;
            const vec4 TvY = closestPointTransform * vY;
            const vec4 TvZ = closestPointTransform * vZ;

            const vec3 p = closestPointTransformed + closestBondingPointOffset.xyz();
            const mat4 pTransform = mat4( TvX, TvY, TvZ, vec4( p, 1.0f ) );

            // mutex is locked, only during add... math happens outside, nice
            AnchorParticle( ivec3( p ), pTransform );

            // since we bound to a surface, the point location is no longer valid and should not proceed
            RespawnParticle( idx );
        }
    }
}
//=================================================================================================
// this is the master thread over the worker threads on the crystal object
inline void Crystal::MonitorThreadFunction () {
    // enter a loop
    while ( !threadKill ) {
        // how do we quit? something indicated from the master
            // we will need a number of atomic signals... screenshot(atomic) + config, quit(atomic),
            // reset(atomic)

        // triggering a screenshot

        // what do we monitor?
            // some kind of flag we watch to indicate master wants a screenshot
                // this should include render config so that animations etc can be orchestrated there
            // when you want to do a screenshot, if you need to prep data, do it here before cutting
                // over to the worker threads to do the work
            // some kind of percentage completion, so that we can report that at the higher level


    // report percentage of the storage pool used, unless a screenshot is running... then, report pixels complete
        // percentageCompletion = float( particleStorageAllocator ) / float( simConfig.numParticlesStorage );
        // percentageCompletion += 0.01f;

        // wait for the next iteration
        sleep_for( 100ms );
        // ssDispatch.store( 0 );
        // while ( ssDispatch < numPixels ) {
            // sleep_for( 1s );
        // }
        
    }
    // when we drop out, indicate termination to the other threads
    threadKill = true; // belt and suspenders...
}
//=================================================================================================
// worker thread, doing the particle update
inline void Crystal::WorkerThreadFunction ( int id ) {
    // enter a loop...
    while ( !threadKill ) {
        /*  If we hit the screenshot counter and get a value less than the number of pixels, we need
        to do work to render an image, instead of the particle update...

            We also need to prevent running out of pointers in the storage pool... so check the
        allocator to see if we are "close" to the maximum, we need to enter a waiting state...
        */

        // do we want screenshot work? if so, where?
        uint32_t x, y;
        const bool ss = ScreenshotIndicated( x, y );

        // otherwise let's do particle work
        uintmax_t i;
        const bool work = ParticleUpdateIndicated( i );

        if ( !ss && !work ) {
            // there is no work to do right now
            sleep_for( 1ms );
        } else if ( ss ) {
        // if ( ss ) {
            // we have secured work for one pixel
            DrawPixel( x, y );
            ++ssComplete; // helps with reporting "percentage complete"
        } else if ( work ) {
        // } else {
            // we need to do work for one particle update, index indicated by the job counter i
            UpdateParticle( i % simConfig.numParticlesScratch );
        }
    }
}
//=================================================================================================