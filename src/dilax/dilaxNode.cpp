#include "dilaxNode.h"
#include <iostream>
#include <mutex>
using namespace std;

namespace dilax_auxiliary {
    std::vector<fan2Leaf*> empty_fan2leaves;
    std::vector<dilaxNode*> empty_fan2nodes;
    std::vector<dilaxNode*> empty_nodes;
    
    // Thread-local storage for retrain buffers to avoid data races
    thread_local keyType *retrain_keys = nullptr;
    thread_local recordPtr *retrain_ptrs = nullptr;
    
    // Mutex to protect global initialization/cleanup
    std::mutex global_aux_mutex;

    void init_insert_aux_vars() {
        // Thread-local initialization - each thread gets its own buffers
        if (retrain_keys) { 
            delete[] retrain_keys; 
        }
        if (retrain_ptrs) { 
            delete[] retrain_ptrs; 
        }
        retrain_keys = new keyType[LEAF_MAX_CAPACIY * 2];
        retrain_ptrs = new recordPtr[LEAF_MAX_CAPACIY * 2];
    }

    void free_insert_aux_vars() {
        // Thread-local cleanup
        if (retrain_keys) {
            delete[] retrain_keys;
            retrain_keys = nullptr;
        }
        if (retrain_ptrs) {
            delete[] retrain_ptrs;
            retrain_ptrs = nullptr;
        }
    }

    // Thread-safe initialization check for each thread
    void ensure_thread_aux_vars() {
        if (!retrain_keys || !retrain_ptrs) {
            init_insert_aux_vars();
        }
    }
}

