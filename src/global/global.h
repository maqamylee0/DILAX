#include <vector>
#include <iostream>
#include <atomic>
#include "global_typedef.h"

#ifndef DILAX_GLOBAL_H
#define DILAX_GLOBAL_H

#define LR_PRED(a, b, key, fanout) std::min<int>(std::max<int>(static_cast<int>(a + b * key), 0), fanout - 1)
//inline int LR_PRED(double a, double b, const long &key, int fanout) { return std::min<int>(std::max<int>(a + b * key, 0), fanout - 1); }
#define MIN_INT(a, b) std::min<int>(a, b)
#define MAX_INT(a, b) std::max<int>(a, b)

#define MIN_LONG(a, b) std::min<long>(a, b)
#define MAX_LONG(a, b) std::max<long>(a, b)

#define MIN_DOUBLE(a, b) std::min<double>(a, b)
#define MAX_DOUBLE(a, b) std::max<double>(a, b)


extern long dilax_totalDataSize;
extern long dilax_halfN;
extern long dilax_query_step;
extern long dilax_query_start_idx;

extern const long dilax_n_query_keys;
extern const double dilax_R1;
extern const double dilax_R2;
extern const double dilax_R3;

//extern const int minFan;
extern const int dilax_fanThreashold;

#define LEAF_MAX_CAPACIY 8192
#define minFan 2

#define MIN_KEY(a, b) std::min<keyType>(a, b)

//extern const int Delta;
//extern const int nThreads;
//extern const int one_in_n;
//extern const double RATIO;
//extern const int maxFan;
//extern const int minFanforSplit;


extern double dilax_RHO;
extern int dilax_buMinFan;
extern double dilax_max_expanding_ratio;
extern double dilax_retrain_threshold;

extern std::atomic<long> dilax_num_adjust_stats;


struct dilaxNode;
struct fan2Leaf;

struct DilaxSearchBound {
    size_t start;
    size_t stop;
};

struct dilaxPairEntry {
    keyType key; // key >= 0: ptr is the index of the record in the data array, key == -1: ptr is a child; key < - 1: this position is empty
    union {recordPtr ptr; dilaxNode *child; fan2Leaf *fan2child; };

    dilaxPairEntry(): key(-3), ptr(-3) {}
    inline void setNull() { key = -3; ptr = -3;}


//    inline long getPayload() { return payload;}
//    inline dilaxNode* getChild() { return child;}
//    inline dilaxfan2Leaf* getFan2Child() { return fan2child;}

    inline void assign(keyType _k, recordPtr _p) { key = _k; ptr = _p; }
    inline void setChild(dilaxNode *_child) { key = -1; child = _child;};
    inline void setFan2Child(fan2Leaf *_fan2child) { key = -2; fan2child = _fan2child;};

};


#endif //DILAX_GLOBAL_H
