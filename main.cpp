//=================================================================================================
// command line output
#include <iostream>
using std::cout;
using std::endl;
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
//=================================================================================================
// mutex / lock stuff
#include <mutex>
using std::mutex;
using std::lock_guard;
//=================================================================================================
// thread stuff
#include <thread>
using std::thread;
//=================================================================================================
// shared_ptr
#include <memory>
using std::make_unique;
using std::unique_ptr;
//=================================================================================================
// specific width types + atomics, for "program counter"
#include <cstdint>
#include <atomic>
using std::atomic_uintmax_t;
//=================================================================================================
// Terminal UI output
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/color.hpp>
using namespace ftxui;
//=================================================================================================
// math/vector stuff
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
// the list of particles which have been anchored
unordered_map< ivec3, vector< unique_ptr< mat4 > > > anchoredParticles;
mutex anchoredParticlesGuard;

// going to use this as a threshold for a kind of "bonding affinity"... may need more detail here than 1 degree of freedom
// float transformSimilarity ( const particle_t &a, const particle_t &b ) const {
    // applying the transform, taking advantage of homogeneous coordinates to only get the orientation of the transform
    // vec3 base = vec3( 1.0f, 0.0f, 0.0f );
    // vec3 tA = ( a.transform * vec4( base, 0.0f ) ).xyz();
    // vec3 tB = ( b.transform * vec4( base, 0.0f ) ).xyz();
    // this should probably get a remapping that peaks at 0 and falls off sharply, like the cauchy thing
    // return tan( dot( tA, tB ) * pi * 2.0f );
// }

vector< vec3 > particlePool;
ivec3 minExtents = ivec3( 0 );
ivec3 maxExtents = ivec3( 0 );

void anchorParticle ( vec3 p ) {
    // we need to anchor this particle... but first, lock the mutex so only one thread can do this at once
    lock_guard< mutex > lock( anchoredParticlesGuard ); // mutex entering scope locks 

    // precompute integer version
    const ivec3 iP = ivec3( p );

    // we need to update our bounds with the new particle location
    minExtents = glm::min( iP, minExtents );
    maxExtents = glm::max( iP, maxExtents );

    // and add the particle... right now we will use only position, but orientation is important for crystal lattice
    mat4 pTransform = glm::translate( mat4( 1.0f ), p );
    anchoredParticles[ iP ].push_back( make_unique< mat4 >( pTransform ) );
}

void reporterThreadFunction () {
    
}

//=================================================================================================
constexpr int NUM_THREADS = 69;
int main () {

	atomic_uintmax_t jobCounter;

    // idea is basically that we're going to have many little oriented crystal elements
    // and they bond to a nearby anchored ball with probability X... when they do, they
    // have a small chance to randomize their orientation, to create defects in the
    // otherwise regular crystal lattice. I think something inspired by the cauchy thing
    // would make a lot of sense, I can look at how it's shaped... but I want a strong
    // chance of a very small jitter and very, very rarely, it's a more significant
    // alteration

    bool threadFences[ NUM_THREADS ] = { false };
	std::thread threads[ NUM_THREADS + 1 ];                   // create thread pool
	for ( int id = 0; id <= NUM_THREADS; id++ ) {            // do work
		threads[ id ] = ( id == NUM_THREADS ) ? std::thread(// reporter thread
			[ = ] () {
				const auto tstart = std::chrono::high_resolution_clock::now();

                return;
            }
		) : std::thread(
			[ id ] () {
                // this thread knows its id...
                cout << NUM_THREADS << " " << id << endl;
                anchorParticle( vec3( id ) );
                return;
			}
		);
	}

    for ( auto& [ cell, ptrVec ] : anchoredParticles ) {
        if ( ptrVec.size() != 0 ) {
            mat4 mat = *ptrVec[ 0 ].get();
            cout << "found anchored" << endl << glm::to_string( mat ) << endl;
        }
    }

	return 0;
}