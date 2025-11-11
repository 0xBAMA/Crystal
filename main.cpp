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
// for dispatching particle updates
atomic_uintmax_t jobCounter { 0 };
const uintmax_t maxParticles = 35'000'000;
const uintmax_t particlesPerStep = 100'000;

// threadpool setup
constexpr int NUM_THREADS = 72;
bool threadFences[ NUM_THREADS ];
bool threadKill;
std::thread threads[ NUM_THREADS ];
//=================================================================================================
// shared_ptr
#include <memory>
using std::make_shared;
using std::shared_ptr;
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
			return h1 ^ ( h2 << 10 ) ^ ( h3 << 20 );
		}
	};
}



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

const int pointerAllocatorChunk = 1000000;
vector< shared_ptr< mat4 > > pointerPool;
atomic_uintmax_t pointerPoolAllocator;
mutex pointerPoolGuard;
uintmax_t getPointerIndex () {
    {
        lock_guard< mutex > lock( pointerPoolGuard );
        if ( pointerPool.size() - pointerPoolAllocator < pointerAllocatorChunk ) {
            pointerPool.resize( pointerPool.size() + pointerAllocatorChunk );
        }
    }
    return pointerPoolAllocator.fetch_add( 1 );
}

struct gridCell {
    vector< shared_ptr< mat4 > > particles;
    std::shared_mutex mutex;

    gridCell() {
        particles.resize( 0 );
        particles.reserve( 128 );
    }

    size_t GetCount () {
        // this is the non-exclusive read mutex
        std::shared_lock lock( mutex );
        return particles.size();
    }

    void Add( const mat4 &pTransform ) {
        // add it to the list... exclusive mutex is locked at this time
        std::unique_lock lock( mutex );
        if ( particles.size() < 128 ) {
            // get a new pointer...
            uintmax_t ptrIdx = getPointerIndex();

            // using the input argument, fill out the data
            shared_ptr< mat4 > ptr = pointerPool[ ptrIdx ];
            *ptr = pTransform;

            // pushing the pointer onto the list
            particles.push_back( ptr );
        }
    }

    shared_ptr< mat4 > Get( const int &idx ) {
        // locking the non-exclusive read mutex
        std::shared_lock lock( mutex );
        return particles[ idx ];
    }
};

// the list of particles which have been anchored
CTSL::HashMap< ivec3, std::shared_ptr< gridCell > > anchoredParticles;

// maintaining stats on the above container
ivec3 minExtents = ivec3( -1 );
ivec3 maxExtents = ivec3( 1 );
int numAnchored = 0;

// get a point on the boundary
thread_local rng pick( 0.0f, 1.0f );
float temperature = 2.0f;
thread_local rngN jitter( 0.0f, 0.1f );

void respawnParticle ( vec4 &p ) {
    const bool face = ( pick() < 0.5f );
    // const bool face = true;
    constexpr float margin = 0.0f;
    p.x = glm::mix( minExtents.x - margin, maxExtents.x + margin, pick() );
    p.y = glm::mix( minExtents.y - margin, maxExtents.y + margin, pick() );
    p.z = glm::mix( minExtents.z - margin, maxExtents.z + margin, pick() );

    switch ( int( std::floor( pick() * 3.0f ) ) ) {
    // switch ( 0 ) {
        // we will be on one of the parallel x faces, flatten
        case 0: p.x = face ? minExtents.x - margin : maxExtents.x + margin; break;

        // ditto, y faces
        case 1: p.y = face ? minExtents.y - margin : maxExtents.y + margin; break;

        // z faces
        case 2: p.z = face ? minExtents.z - margin : maxExtents.z + margin; break;

        // won't hit much (can use this intentionally), but will be uniform random spawn if you do
        default:
            break;
    }

    // the .w will be a counter... it is decremented any update during which it is outide the current bounding volume
    p.w = 69.0f; // I think this is a nice mechanism - it can wander, but in a bounded way - when it hits zero, respawn
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

    // move the particle slightly... small change each update
    constexpr vec3 staticFlow = 0.0f * vec3( 1.0f, 3.0f, 1.0f );
    thread_local vec4 flowVector = vec4( ( 1.0f + 0.3f * pick() ) * glm::normalize( vec3( jitter(), jitter(), jitter() ) ), 0.0f );
    flowVector = glm::rotate( identity, jitter(), glm::normalize( vec3( jitter(), jitter(), jitter() ) ) ) * v0;
    particle += vec4( temperature * vec3( jitter(), jitter(), jitter() ) + pick() * flowVector.xyz() + pick() * staticFlow, 0.0f );

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
                nearbyPoints.push_back( *gcp->Get( i ) ) ;
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
        if ( closestPointDistance < 0.5f ) { // close enough... random hash... etc
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
            if ( pick() < 0.00001f ) {
                closestBondingPointOffset += 10.0f * vec3( jitter(), jitter(), jitter() );
                closestPointTransform = glm::translate( glm::rotate( glm::translate( closestPointTransform, -closestPointTransformed ), 100.0f * jitter(), glm::normalize( vec3( jitter(), jitter(), jitter() ) ) ), closestPointTransformed );
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
int gifFrame = 0;
const int gifDelay = 4;
const int imageWidth = 1280;
const int imageHeight = 720;
vector< uint8_t > screenshotBufferData;

// operating on a buffer of image data
thread_local vector< float > scratch; // I'd like to move to something consolidated/straightforward like this

thread_local int binCountsA[ imageWidth * imageHeight ] = { 0 };
thread_local float binHeights[ imageWidth * imageHeight ] = { 0 };

vec3 color1 = vec3( 1.0f, 0.0f, 0.0f );
vec3 color2 = vec3( 0.5f );

void prepareOutputFrame ( vector< uint8_t > &data, const int count, const vec3 minExtentsIn, const vec3 maxExtentsIn, mat4 transform ) {
// trying to do something with some smoothing... need to precompute bounding boxes for each frame if I'm going to do this
    // static ivec3 displayExtentsMin = minExtents - ivec3( 5 );
    // displayExtentsMin = ivec3( glm::mix( vec3( displayExtentsMin ), vec3( minExtents ) - vec3( 5.0f ), 0.5f ) );

    // static ivec3 displayExtentsMax = maxExtents + ivec3( 5 );
    // displayExtentsMax = ivec3( glm::mix( vec3( displayExtentsMax ), vec3( maxExtents ) + vec3( 5.0f ), 0.5f ) );

    // static ivec3 midPoint = ( displayExtentsMin + displayExtentsMax ) / 2;
    // midPoint = ivec3( glm::mix( vec3( midPoint ), vec3( displayExtentsMin + displayExtentsMax ) / 2.0f, 0.5f ) );

    int margin = -10;
    vec3 displayExtentsMin = minExtentsIn - vec3( margin );
    vec3 displayExtentsMax = maxExtentsIn + vec3( margin );
    const vec3 midPoint = vec3( displayExtentsMin + displayExtentsMax ) / 2.0f;

    const vec3 spans = maxExtentsIn - minExtentsIn;
    const float maxSpan = max( max( spans.x, spans.y ), spans.z );
    const vec3 delta = vec3( maxSpan ) - spans;
    displayExtentsMin -= delta;
    displayExtentsMax += delta;
    
    // restore prior values of z for color stuff
    displayExtentsMin.z = minExtentsIn.z;
    displayExtentsMax.z = maxExtentsIn.z;

    for ( auto& b : binCountsA ) { b = 0; }
    for ( auto& b : binHeights ) { b = 0; }

    // thread_local float binLerpV[ imageWidth * imageHeight ] = { 0 };

    int maxCountA = 0;
    int nonzeroBinsA = 0;
    for ( uintmax_t i = 0; i < count; i++ ) {
        const vec3 p = ( *pointerPool[ i ] * p0 ).xyz();
        const vec3 pT = glm::translate( transform * glm::translate( identity, -vec3( midPoint ) ), vec3( midPoint ) ) * vec4( p, 1.0f );
        const vec3 loc = vec3(
            remap( pT.x, displayExtentsMin.x, displayExtentsMax.x, 0.0f, imageWidth * ( float( imageHeight ) / float( imageWidth ) ) ) + imageWidth / 4.0f,
            remap( pT.y, displayExtentsMin.y, displayExtentsMax.y, 0.0f, imageHeight ),
            remap( pT.z, displayExtentsMin.z, displayExtentsMax.z, 0.0f, 1000.0f )
        );

        constexpr float ratio = ( 21.0f / 9.0f );
        const vec2 uv = ( loc.xy() ) / vec2( imageWidth, imageHeight );

        if ( glm::all( glm::lessThan( uv, vec2( 1, 1.0f - 0.5f / ratio ) ) ) &&
            glm::all( glm::greaterThan( uv, vec2( 0, 0.5f / ratio ) ) ) ) {

            const int idx = int( loc.x ) + imageWidth * int( loc.y ); 
            binCountsA[ idx ]++;
            // binHeights[ idx ] = max( clamp( remap( p.z, displayExtentsMin.z, displayExtentsMax.z, 0.0f, 1000.0f ), 0.0f, 999.0f ), binHeights[ idx ] );
            binHeights[ idx ] = max( float( i ) / float( count ), binHeights[ idx ] );
            maxCountA = std::min( std::max( maxCountA, binCountsA[ idx ] ), int( 512 ) );
        }
    }

    // cout << "processed " << ptrCache << " particles" << endl;
    // cout << "found max " << maxCountA << " points per screen bin" << endl;

    // write out the image
    // data.clear();
    // data.reserve( 4 * imageWidth * imageHeight );

    for ( auto& c : data ) {
        c = 0;
    }

    for ( int i = 0; i < ( imageWidth * imageHeight * 4 ); i += 4 ) {
        // some height term and a density term
        float h = binHeights[ i / 4 ];

        // float b = ( binCountsA[ i ] == 0 ) ? 0 : ( 255 * glm::exp( -0.001f * float( binCountsA[ i ] ) / float( maxCountA ) ) );
        float b = 255 * glm::pow( float( binCountsA[ i / 4 ] )  / float( maxCountA ), 0.8f );        

        vec3 c = glm::mix( vec3( 1.0f ), vec3( 0.5f, 0.0f, 1.0f ), vec3( sqrt( h ) ) );

        data[ i + 0 ] = ( std::clamp( uint( b * c.x ), 0u, 255u ) );
        data[ i + 1 ] = ( std::clamp( uint( b * c.y ), 0u, 255u ) );
        data[ i + 2 ] = ( std::clamp( uint( b * c.z ), 0u, 255u ) );
        data[ i + 3 ] = ( 255 );
    }
}

static int maxCount = 0;
static CTSL::HashMap< ivec3, vec4 > voxelModel;
void prepareOutputFrameDeltaTracking ( bool rebuildMap, vector< uint8_t > &data, const int count, const vec3 minExtentsIn, const vec3 maxExtentsIn, mat4 transform, vec3 lightDirection ) {

    /// from wrighter:
    /*
        vec4 trace ( inout vec3 p, vec3 rd ) {
            float dist = 0.0f;
            float mv_dens = 40.0f; // max volume density
            float sd = -1.0f;
            for( float i = 0.0f; i < 50.; i++){
                float t = -log(1.-hash_f()) / mv_dens; // delta tracking

                //if(gl_FragCoord.x>iResolution.x/2.) // sanity check, cuz sdfs can be weird
                t = max(t,sd); // speed up empty space (optional)

                dist += t;
                p += rd * t;

                sd = map(p, iTime).x;

                float dens = 0.5;
                bool hit = dens > hash_f();
                if(hit){
                    return vec4(material);
                }
            }
            return vec4(-1);
        }
    */

    if ( rebuildMap ) {
    // prepare the hashmap... we only need color + density
        cout << "constructing frame hashmap " << endl;
        voxelModel.clear();
        maxCount = 0;
        
        ivec3 minExtentComputed = ivec3( 1000 );
        ivec3 maxExtentComputed = ivec3( -1000 );

        for ( int i = 0; i < count; i++ ) {
            shared_ptr< mat4 > ptr = pointerPool[ i ];
            vec4 p = *ptr * p0;
            ivec3 iP = ivec3( p.xyz() );

            vec4 temp;
            vec4 col = vec4( glm::mix( color1, color2, float( i ) / float( count ) ), 1.0f );
            if ( voxelModel.find( iP, temp ) ) {
                temp += col;
            } else {
                temp = col;
            }

            minExtentComputed = min( minExtentComputed, iP );
            maxExtentComputed = max( maxExtentComputed, iP );
            maxCount = max( int( temp.a ), maxCount );
            voxelModel.insert( iP, temp );
        }

        // normalizing...
        for ( int x = -minExtentComputed.x; x <= maxExtentComputed.x; x++ ) {
            for ( int y = -minExtentComputed.y; y <= maxExtentComputed.y; y++ ) {
                for ( int z = -minExtentComputed.z; z <= maxExtentComputed.z; z++ ) {
                    vec4 temp;
                    const ivec3 p = ivec3( x, y, z );
                    if ( voxelModel.find( p, temp ) ) {
                        temp.x = temp.x / temp.w; // averaged color
                        temp.y = temp.y / temp.w;
                        temp.z = temp.z / temp.w;
                        temp.w = pow( temp.w / maxCount, 1.0f ); // normalized density
                        voxelModel.insert( p, temp );
                    }
                }
            }
        }
        cout << "done." << endl;
    }

    const vec3 midPoint = ( minExtentsIn + maxExtentsIn ) / 2.0f;

    thread renderThreads[ NUM_THREADS ];
    atomic_uintmax_t renderThreadDispatch{ 0 };

    const int numPixels = imageWidth * imageHeight;
    for ( auto& t : threads ) {
        t = thread( [&] () {
            while ( true ) {
                int pixel = renderThreadDispatch.fetch_add( 1 );
                if ( pixel < numPixels ) {
                    // render a pixel
                    const int x = pixel % imageWidth;
                    const int y = pixel / imageWidth;

                    const int idx = 4 * pixel;

                    // black bar check
                    constexpr float ratio = ( 21.0f / 9.0f );
                    if ( const vec2 uv = vec2( x + 0.5f, y + 0.5f ) / vec2( imageWidth, imageHeight );
                    glm::all( glm::lessThan( uv, vec2( 1, 1.0f - 0.5f / ratio ) ) ) &&
                    glm::all( glm::greaterThan( uv, vec2( 0, 0.5f / ratio ) ) ) ) {

                        // you are inside the image... do the delta tracking raymarch
                        vec3 color = vec3( 0.0f );

                        // camera setup
                            // origin, direction
                        vec2 uvAdjust = ( uv - vec2( 0.5f, 0.5f ) ) * vec2( 1.0f, float( imageHeight ) / float( imageWidth ) );
                        mat4 inverseTransform = glm::inverse( transform );
                        vec3 rO = inverseTransform * vec4( vec3( 200.0f * uvAdjust, -40 ), 1.0f );
                        vec3 rD = inverseTransform * vec4( 0.0f, 0.0f, 1.0f, 0.0f );

                        // track the ray from the eye through the volume...
                        int maxDistance = std::ceil( glm::distance( minExtentsIn, maxExtentsIn ) );
                        float tMin, tMax;
                        if ( IntersectAABB( rO, rD, minExtentsIn, maxExtentsIn, tMin, tMax ) ) {
                            // color = vec3( 1.0f, 0.0f, 0.0f );
                            // vec3 p = rO + max( 0.0f, tMin ) * rD;
                            vec3 p = rO + tMin * rD;
                            constexpr int samples = 128;
                            for ( int s = 0; s < samples; s++ ) {
                                for ( int i = 0; i < maxDistance; i++ ) {
                                    float t = -log( pick() );
                                    p += t * rD;

                                    if ( glm::any( glm::lessThanEqual( p, vec3( minExtentsIn ) ) ) ||
                                        glm::any( glm::greaterThanEqual( p, vec3( maxExtentsIn ) ) ) ) {
                                        // oob
                                        break;
                                    }

                                    vec4 temp;
                                    if ( voxelModel.find( ivec3( p ), temp ) ) {
                                        if ( temp.a > pick() ) { // this is the hit condition...
                                            // do we hit something, going up?
                                            vec3 pShadow = p;
                                            float shadowTerm = 1.0f;

                                            for ( int j = 0; j < 50; j++ ) {
                                                pShadow += lightDirection * float( -log( pick() ) );
                                                vec4 tempShadow;
                                                if ( voxelModel.find( ivec3( pShadow ), tempShadow ) ) {
                                                    if  ( temp.a > pick() || j > 195 ) {
                                                        shadowTerm = 0.1f;
                                                    }
                                                }
                                            }

                                            color += vec3( temp.xyz() * shadowTerm ) / float( 4 * samples );
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        data[ idx + 0 ] = static_cast< uint8_t >( clamp( color.r * 255.0f, 0.0f, 255.0f ) );
                        data[ idx + 1 ] = static_cast< uint8_t >( clamp( color.g * 255.0f, 0.0f, 255.0f ) );
                        data[ idx + 2 ] = static_cast< uint8_t >( clamp( color.b * 255.0f, 0.0f, 255.0f ) );
                        data[ idx + 3 ] = 255;

                    } else {
                        // you are inside the black area, color black
                        for ( int i = 0; i < 3; i++ ) {
                            data[ idx + i ] = 0;
                        }
                        data[ idx + 3 ] = 255;
                    }
                } else {
                    break;
                }
            }
        });
    }

    for ( auto& t : threads ) {
        t.join();
    }
}

void prepareOutputScreenshotDeltaTracking () {
    prepareOutputFrameDeltaTracking( true, screenshotBufferData, pointerPoolAllocator, minExtents, maxExtents, glm::rotate( identity, 0.5f, glm::normalize( vec3( 0.7f, 1.0f, 0.8f ) ) ), normalize( vec3( 1.0f ) ) );

    auto now = std::chrono::system_clock::now();
    auto inTime_t = std::chrono::system_clock::to_time_t( now );
    std::stringstream ssA;
    ssA << std::put_time( std::localtime( &inTime_t ), "Crystal-%Y-%m-%d at %H-%M-%S.png" );
    stbi_write_png( ssA.str().c_str(), imageWidth, imageHeight, 4, &screenshotBufferData[ 0 ], 4 * imageWidth );
}

void prepareOutputScreenshot () {
    // create a linear buffer of all the point locations in minimum representation... this has a lot of room for optimization...
        // writing an allocating layer out of a pool of shared_ptrs() may be faster? and will allow me to actually encode the
        // specific order in which all the particles were bound to the crystal. They will be allocated in the order they bond,
        // which means that this does a couple things: first, I can playback the process... second, I have this information
        // precomputed, so that we don't have to do this iteration over the hashmap first.
    prepareOutputFrame( screenshotBufferData, pointerPoolAllocator, minExtents, maxExtents, identity );

	auto now = std::chrono::system_clock::now();
	auto inTime_t = std::chrono::system_clock::to_time_t( now );
	std::stringstream ssA;
	ssA << std::put_time( std::localtime( &inTime_t ), "Crystal-%Y-%m-%d at %H-%M-%S.png" );
    stbi_write_png( ssA.str().c_str(), imageWidth, imageHeight, 4, &screenshotBufferData[ 0 ], 4 * imageWidth );
}



//=================================================================================================
#include "reporter.h" // proc filesystem reading
//=================================================================================================
int main () {
    // pump initial proc data
    updateProcData();
    updateProcData();

    // allocate space for the screenshot
    screenshotBufferData.resize( imageWidth * imageHeight * 4 );

    // new random colors
    color1 = vec3( 1.0f );
    // color2 = vec3( 0.8f * pick() + 0.1f, 0.3f * pick() + 0.7f, 0.5f * pick() + 0.5f );
    color2 = vec3( 1.0f, 0.5f + 0.5f * pick(), 0.3f + 0.7f * pick() );

    // setting initial program state
    threadKill = false;
    // set all the thread fences "true"
    for ( auto& fence : threadFences )
        fence = true;

    // we need to make sure there are particles to update...
    cout << "Spawning Particles... ";
    for ( auto& p : particlePool ) {
        respawnParticle( p );
    }
    cout << "Done." << endl;

    cout << "Spawning Pointer Pool... ";
    for ( int i = 0; i < maxParticles + 100000; i++ ) {
        pointerPool.push_back( make_shared< mat4 >() );        
    }
    cout << "Done." << endl;

    // an initial point in the model, so we have something to bond to
    cout << "Anchoring Initial Seed Particles... ";
    rng pR( -2.0f, 2.0f );
    for ( int i = 0; i < 20; i++ ) {
        mat4 transform = glm::rotate( glm::translate( identity, 10.0f * vec3( 3.0f * ( 2.0f * pick() - 1.0f ), ( 2.0f * pick() - 1.0f ), pick() ) ), 10.0f * pR(), glm::normalize( vec3( jitter(), jitter(), jitter() ) ) );
        anchorParticle( transform * p0, transform );
    }
    cout << "Done." << endl;

    // we need to pick a set of offsets to use for the crystal growth...
        // start with a "tetragonal", which will emphasize orientation because it has longer offsets along one axis
    // the angle is uniform, so we will leave this as just a nonuniformly scaled basis...
    const float xXx = 0.1618f + 0.3f * pick();
    const float yYy = 0.618f + 0.1f * jitter();
    const float zZz = 0.618f;

    const float pi = 3.1415926535f;
    const mat4 rX = glm::rotate( identity, 0.0f + pick(), normalize( vec3( 0.0f, 0.0f, 1.0f ) ) );
    const mat4 rY = glm::rotate( identity, pick() * pi / 2.0f, normalize( vec3( 0.0f, 0.0f, 1.0f ) ) );
    const mat4 rZ = glm::rotate( identity, pi / 5.0f, normalize( vec3( 1.0f, 0.0f, 0.0f ) ) );

    // note that 6 bonding points is in no way a constraint here
    bondingSiteOffsets = {
        // rZ * vec4( 0.0f, 0.0f, -zZz, 0.0f ),
        rZ * vec4( 0.0f, 0.0f, zZz, 0.0f ),
        // rY * vec4( 0.0f, -yYy, 0.0f, 0.0f ),
        rY * vec4( 0.0f, yYy, 0.0f, 0.0f ),
        // rX * vec4( -xXx, 0.0f, 0.0f, 0.0f ),
        rX * vec4( xXx, 0.0f, 0.0f, 0.0f ),
    };
    
    cout << "Spawning Reporter Thread.......... ";
	std::thread reporterThread = std::thread(
        [&] () { // this is the reporter thread
			auto tStart = high_resolution_clock::now();
            cout << "Done." << endl;

            size_t update = 0;

            int window_1_left = 1;
            int window_1_top = 8;
            int window_1_width = 36;
            int window_1_height = 14;

            bool checked[3] = {false, false, false};

            auto window_1 = Window({
                .inner = Container::Vertical({
                    // Checkbox("Check me", &checked[0]),
                    // Checkbox("Check me", &checked[1]),
                    // Checkbox("Check me", &checked[2]),
                    Slider( "Temperature", &temperature, 0.0f, 69.0f ),
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
                            hbox({ text( "Frames Queued: " ) | c1, text( to_string( pointerPoolAllocator / particlesPerStep ) ) | c2 }),
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
                .inner = Renderer([] ( bool focused ) {
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

            auto screen = ScreenInteractive::FixedSize( 36, 18 );
            auto ftxDAG = CatchEvent( window_container, [&]( Event event ) {
                if ( event == Event::Character('q') ) {
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
                screen.RequestAnimationFrame();
                loop.RunOnce();
                sleep_for( 100ms ); 

                if ( numAnchored >= maxParticles ) {                   
                    quit = true;
                }
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
                        sleep_for( 1ms );
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

    int frames = pointerPoolAllocator / particlesPerStep;
    cout << " Rendering Animation Consisting of " << frames << " frames" << endl;
    // atomic_uintmax_t frameDispatcher{ 0 };
    
    // // the bulk data storage vector
    // vector< vector< uint8_t > > frameData;
    vec3 pInit = ( *pointerPool[ 0 ] * p0 ).xyz();
    vector< vec3 > minExtentsData; vec3 runningMin = pInit;
    vector< vec3 > maxExtentsData; vec3 runningMax = pInit;
    
    for ( int i = 0; i <= frames; i++ ) {
    //     frameData.emplace_back( 4 * imageWidth * imageHeight );
        for ( int j = 0; j < particlesPerStep; j++ ) {
            const int idx = std::min( int( i * particlesPerStep + j ), int( numAnchored ) );
            const vec3 p = ( *pointerPool[ idx ] * p0 ).xyz();
            runningMin = glm::min( runningMin, p );
            runningMax = glm::max( runningMax, p );
        }
        minExtentsData.emplace_back( runningMin );
        maxExtentsData.emplace_back( runningMax );
    }
    minExtentsData.emplace_back( runningMin );
    maxExtentsData.emplace_back( runningMax );

    // cout << "Allocated space:" << endl;
    // for ( auto& frame : frameData ) {
        // cout << frame.size();
    // }

    /*
    cout << "starting..." << endl;
    thread renderThreads[ NUM_THREADS + 1 ];
    for ( int i = 0; i < NUM_THREADS + 1; i++ ) {
        renderThreads[ i ] = ( i == 0 ) ? thread( [&] () {
            // reporter thread
            while ( true ) {
                sleep_for( 1000ms );
                cout << "Dispatched " << frameDispatcher << " frames... ( of " << frames << " )" << endl;
                if ( frameDispatcher > frames ) {
                    break;
                }
            }
        } ) : thread ( [&] () {
            // worker thread
            while ( true ) {
                int frame = frameDispatcher.fetch_add( 1 );
                if ( frame > frames ) {                    
                    break;
                } else {
                    // render a frame...
                    prepareOutputFrame( frameData[ frame ], particlesPerStep * frame,
                        glm::mix( minExtentsData[ frame ], minExtentsData[ frame + 1 ], 0.5f ),
                        glm::mix( maxExtentsData[ frame ], maxExtentsData[ frame + 1 ], 0.5f ),
                        glm::scale( glm::rotate( identity, frame * 0.001f, glm::normalize( vec3( 0.7f, 1.0f, 0.8f ) ) ), vec3( glm::mix( 1.2f, 0.8f, float( frame ) / float( frames ) ) ) ) );

                    // report finished
                    cout << "finished frame " << frame << endl;
                }
            }
        } );
    }

    // terminate everyone
    for ( auto& thread : renderThreads ) {
        thread.join();
    }
     */

    GifWriter g;
	auto now = std::chrono::system_clock::now();
	auto inTime_t = std::chrono::system_clock::to_time_t( now );
	std::stringstream ssA;
	ssA << std::put_time( std::localtime( &inTime_t ), "Crystal-%Y-%m-%d at %H-%M-%S.gif" );
	GifBegin( &g, ssA.str().c_str() , imageWidth, imageHeight, gifDelay );

    vector< uint8_t > dataVec;
    dataVec.resize( imageWidth * imageHeight * 4 );
    const uintmax_t ptrAllocateCache = pointerPoolAllocator;
    int frame = frames;
    for ( int i = frames; i < frames + 400; i++ ) {
        pointerPoolAllocator = std::min( uintmax_t( ptrAllocateCache ), uintmax_t( i * particlesPerStep ) );
        int f = std::min( frame, int( frames ) );
        vec3 sunDirection = glm::rotate( identity, frame * 0.01f, vec3( 1.0f, 2.0f, 4.0f ) ) * v0;
        prepareOutputFrameDeltaTracking( ( i <= frames ), dataVec, particlesPerStep * frame,
            glm::mix( minExtentsData[ f ], minExtentsData[ f + 1 ], 0.5f ),
            glm::mix( maxExtentsData[ f], maxExtentsData[ f + 1 ], 0.5f ),
            glm::scale( glm::rotate( identity, frame * 0.003f, glm::normalize( vec3( 0.7f, 1.0f, 0.8f ) ) ),
                vec3( ( 2.0f / distance( glm::mix( minExtentsData[ f ], minExtentsData[ f + 1 ], 0.5f ), glm::mix( maxExtentsData[ f], maxExtentsData[ f + 1 ], 0.5f ) ) ) *
                glm::mix( 1.1f, 0.9f, glm::smoothstep( float( frame ) / float( frames ), 0.0f, 1.0f ) ) ) ), sunDirection );

        GifWriteFrame( &g, dataVec.data(), imageWidth, imageHeight, gifDelay );
        cout << "finished frame " << frame << " / " << frames + 400 << endl;
        frame++;
    }
	cout << "Prepping GIF file" << ssA.str() << "..." << std::flush;
    GifEnd( &g );
    cout << endl << "Done." << endl;

    // example "lossless" conversion from gif to mp4
    // ffmpeg -i in.gif -c:v libx264 -preset veryslow -qp 0 output.mp4

    // prepareOutputScreenshot();
    // prepareOutputScreenshotDeltaTracking();

	return 0;
}
