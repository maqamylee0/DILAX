#include "dilaxNode.h"
#include <iostream>
using namespace std;

namespace dilax_auxiliary {
    std::vector<fan2Leaf*> empty_fan2leaves;
    std::vector<dilaxNode*> empty_fan2nodes;
    std::vector<dilaxNode*> empty_nodes;
    keyType *retrain_keys = NULL;
    recordPtr *retrain_ptrs = NULL;

    void init_insert_aux_vars() {
        if (retrain_keys) { delete[] retrain_keys; }
        if (retrain_ptrs) { delete[] retrain_ptrs; }
        retrain_keys = new long[LEAF_MAX_CAPACIY * 2];
        retrain_ptrs = new long[LEAF_MAX_CAPACIY * 2];
    }

    void free_insert_aux_vars() {
        if (retrain_keys) {
            delete[] retrain_keys;
            retrain_keys = NULL;
        }
        if (retrain_ptrs) {
            delete[] retrain_ptrs;
            retrain_ptrs = NULL;
        }
    }
}

