#include "crystal.h"

// master kill for all threads
atomic< bool > threadKill = false;

int main ( int argc, char** argv ) {

    cout << "Hello World" << endl;

    Crystal c;

    cout << "Goodbye World" << endl;

    // Proc Updater Thread
        // make sure the proc header has "previous" data

    // Terminal UI Updater Thread
        // CPU activity
        // other central activity monitoring
        // the active list of crystals
            // option to add
            // generate preview image on a given crystal
            // automatically save out on completion

    // something for rendering animations

}