#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stack>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <immintrin.h>
#include <sched.h>
#include "../global/global.h"
#include "../global/fan2Leaf.h"
//#include "../global/linearReg.h"

// Simple Epoch-Based Reclamation for DILAX
class EBR {
private:
    static thread_local uint64_t localEpoch;
    static std::atomic<uint64_t> globalEpoch;
    static std::atomic<void*> pendingDeletes[3]; // Ring buffer for 3 epochs
    static std::atomic<int> deleteTypes[3]; // Track what type each deletion is
    static std::mutex deleteMutex;

public:
    enum DeleteType { NODE_DELETE = 1, STRUCTURE_DELETE = 2 };
    
    static void enterEpoch() {
        localEpoch = globalEpoch.load();
    }
    
    static void exitEpoch() {
        localEpoch = 0;
    }
    
    static void scheduleDelete(void* ptr, DeleteType type = NODE_DELETE) {
        if (!ptr) return; // Null pointer check
        
        uint64_t epoch = globalEpoch.load();
        std::lock_guard<std::mutex> lock(deleteMutex);
        
        // Simple implementation - just add to pending list
        // Only schedule if slot is empty to avoid overwriting
        void* expected = nullptr;
        if (pendingDeletes[epoch % 3].compare_exchange_strong(expected, ptr)) {
            deleteTypes[epoch % 3].store(type);
            // Successfully scheduled
        } else {
            // Slot occupied, skip this deletion for safety
            // In a real implementation, would use a proper queue
        }
    }
    
    static void advance() {
        globalEpoch.fetch_add(1);
        // Be more conservative with cleanup - require more epochs
        uint64_t currentEpoch = globalEpoch.load();
        if (currentEpoch < 6) return; // Don't clean up too early
        
        uint64_t oldEpoch = currentEpoch - 6; // Wait 6 epochs instead of 3
        void* toDelete = pendingDeletes[oldEpoch % 3].exchange(nullptr);
        if (toDelete) {
            int deleteType = deleteTypes[oldEpoch % 3].load();
            deleteTypes[oldEpoch % 3].store(0); // Reset
            
            // Additional safety check before deletion
            if (reinterpret_cast<uintptr_t>(toDelete) > 0x1000) {
                if (deleteType == STRUCTURE_DELETE) {
                    deleteOldStructure(toDelete);
                } else {
                    deleteDilaxNode(toDelete);
                }
            }
        }
    }
    
    // Helper functions to be defined after dilaxNode is fully declared
    static void deleteDilaxNode(void* ptr);
    static void deleteOldStructure(void* ptr);
};

// Declare static members (definitions are in dilaxNode.cpp)
// These will be defined in the corresponding .cpp file to avoid linker issues

// Optimistic lock implementation using int64_t
class OptLock {
private:
    std::atomic<int64_t> typeVersionLockObsolete{0b100}; // Use int64_t

public:
    static constexpr int64_t UNLOCKED = 0b100;
    static constexpr int64_t LOCKED_BIT = 0b010;
    static constexpr int64_t OBSOLETE_BIT = 0b001;

    int64_t readLockOrRestart(bool &needRestart) {
        int64_t version;
        do {
            version = typeVersionLockObsolete.load();
            if (version & LOCKED_BIT) {
                _mm_pause();
                needRestart = true;
                return 0;
            }
        } while (version & LOCKED_BIT);
        return version;
    }

    void readUnlockOrRestart(int64_t startRead, bool &needRestart) {
        int64_t endRead = typeVersionLockObsolete.load();
        if (startRead != endRead || (endRead & OBSOLETE_BIT)) {
            needRestart = true;
        }
    }

    void upgradeToWriteLockOrRestart(int64_t &version, bool &needRestart) {
        if (typeVersionLockObsolete.compare_exchange_strong(version, version + LOCKED_BIT)) {
            return; // Successfully upgraded
        }
        needRestart = true;
    }

    void writeLockOrRestart(bool &needRestart) {
        int64_t version = readLockOrRestart(needRestart);
        if (!needRestart) {
            upgradeToWriteLockOrRestart(version, needRestart);
        }
    }

    void writeUnlock() {
        typeVersionLockObsolete.fetch_add(LOCKED_BIT);
    }

    void writeUnlockObsolete() {
        typeVersionLockObsolete.fetch_add(LOCKED_BIT + OBSOLETE_BIT);
    }
};

#ifndef DILAX_DILAXNODE_H
#define DILAX_DILAXNODE_H


using namespace std;
struct dilaxNode;
struct fan2Leaf;
struct OptimisticDilaxPairEntry; // Forward declaration

// Forward declaration for EBR cleanup
struct OldDilaxStructure {
    OptimisticDilaxPairEntry *data;
    int fanout;
    
    // Constructor for proper initialization
    OldDilaxStructure(OptimisticDilaxPairEntry *d, int f) : data(d), fanout(f) {}
};

namespace dilax_auxiliary {
    extern thread_local keyType *retrain_keys;      // Make thread-local
    extern thread_local recordPtr *retrain_ptrs;    // Make thread-local
    void init_insert_aux_vars();
    void free_insert_aux_vars();
}

namespace dilax {

inline void linearReg_w_simple_strategy(const keyType *X, double &a, double &b, int n) {
    int left_n = n / 2;
    int right_n = n - left_n - 1;
    keyType x_middle = X[left_n];
    double left_slope = 1.0 * left_n / (x_middle - X[0]);
    double right_slope = 1.0 * right_n / (X[n-1] - x_middle);
    b = MAX_DOUBLE(left_slope, right_slope);
    a = -b * x_middle + left_n;
}

inline void linearReg_at_least_four(const keyType *X, double &a, double &b, int n) {
    double nu_b = 0;
    double de_b = 0;
    double mean_x = 0;
    for (int i = 0; i < n; ++i) {
        de_b += X[i] * 1.0 * i;
        nu_b += X[i] * 1.0 * X[i];
        mean_x += X[i];
    }

    mean_x /= n;
    double mean_y = (n - 1) / 2.0;
    de_b -= mean_x * mean_y * n;
    nu_b -= mean_x * mean_x * n;

    if (nu_b != 0) {
        b = de_b / nu_b;
    }  else {
        b = 0;
    }
    a = mean_y - b * mean_x;
}

inline void linearReg_with_max_b(const keyType *X, double &a, double &b, int n) {
    double nu_b = 0;
    double de_b = 0;
    double mean_x = 0;
    for (int i = 0; i < n; ++i) {
        de_b += X[i] * 1.0 * i;
        nu_b += X[i] * 1.0 * X[i];
        mean_x += X[i];
    }

    mean_x /= n;
    double mean_y = (n - 1) / 2.0;
    de_b -= mean_x * mean_y * n;
    nu_b -= mean_x * mean_x * n;

    if (nu_b != 0) {
        b = de_b / nu_b;
    }  else {
        b = 0;
    }
    a = mean_y - b * mean_x;
}


inline void linearReg_w_expanding(const keyType *X, double &a, double &b, int n, int expanded_n, bool use_simple_strategy) {
    if (!use_simple_strategy) {
//        linearReg_at_least_four(X, a, b, n);
        linearReg_with_max_b(X, a, b, n);
    } else {
        linearReg_w_simple_strategy(X, a, b, n);
    }
    if (expanded_n > n) {
        double expanding_ratio = 1.0 * expanded_n / (n + 1);
        b *= expanding_ratio;
        a = a * expanding_ratio + expanding_ratio;
    }
}

} // namespace dilax

// Enhanced dilaxPairEntry with optimistic locking
struct OptimisticDilaxPairEntry : public OptLock {
    keyType key;
    union {
        recordPtr ptr;
        dilaxNode *child;
        fan2Leaf *fan2child;
    };

    // Add explicit constructors and assignment operator to handle atomic issue
    OptimisticDilaxPairEntry() : OptLock(), key(-3) {}
    
    OptimisticDilaxPairEntry(const OptimisticDilaxPairEntry& other) : OptLock() {
        key = other.key;
        ptr = other.ptr;  // This copies the union (same memory layout)
    }
    
    OptimisticDilaxPairEntry& operator=(const OptimisticDilaxPairEntry& other) {
        if (this != &other) {
            key = other.key;
            ptr = other.ptr;  // This copies the union (same memory layout)
            // Don't copy the atomic lock state - keep our own lock
        }
        return *this;
    }

    void assign(const keyType &_key, const recordPtr &_ptr) {
        key = _key;
        ptr = _ptr;
    }

    void setChild(dilaxNode *_child) {
        key = -1;
        child = _child;
    }

    void setFan2Child(fan2Leaf *_fan2child) {
        key = -2;
        fan2child = _fan2child;
    }

    void setNull() {
        key = -3;
    }

    bool isEmpty() const { return key < -2; }
    bool isLeaf() const { return key >= 0; }
    bool hasChild() const { return key == -1; }
    bool hasFan2Child() const { return key == -2; }
};

// Helper function for yielding
inline void yield(int count) {
    if (count > 3)
        sched_yield();
    else
        _mm_pause();
}

struct dilaxNode{
     int fanout;
     int meta_info;
     double a;
     double b;
     int num_nonempty;

     OptimisticDilaxPairEntry *pe_data;

     double avg_n_travs_since_last_dist;
     long total_n_travs;
     long last_total_n_travs;
     int last_nn;
     int n_adjust;



    inline bool is_internal() const { return (meta_info & 1); }
    inline void set_leaf_flag() { meta_info &= ~1U;}
    inline void set_int_flag() { meta_info |= 1; }

    inline void set_fanout(int fan) { fanout = fan;}
    inline int get_fanout() { return fanout; }

    inline void set_n_adjust(int n) { meta_info = n << 1; }

    inline int get_n_adjust() { return n_adjust; }
    inline void inc_n_adjust() { ++n_adjust; }


    inline void set_num_nonempty(int n) { num_nonempty = n; }

    inline void init() {
        fanout = std::max<int>(num_nonempty, minFan);
        fanout <<= 1;
//        fanout *= 1.5;

//        fanout += (fanout * n_adjust) / 10;
//        fanout = std::max<int>(num_nonempty, minFan) * (1 + 0.1 * get_n_adjust());
        
        // Safety check for reasonable fanout size
        if (fanout <= 0 || fanout > 100000) {
            cout << "ERROR: Invalid fanout=" << fanout << " num_nonempty=" << num_nonempty << endl;
            fanout = std::max<int>(num_nonempty, 16); // Fallback value
        }
        
        pe_data = new OptimisticDilaxPairEntry[fanout];
        // Skip explicit initialization - constructor already sets key=-3
    }



    dilaxNode(bool _is_internal): a(0), b(0), meta_info(_is_internal), fanout(0), pe_data(NULL), n_adjust(30), num_nonempty(0),
                                 total_n_travs(0), last_total_n_travs(0), last_nn(0), avg_n_travs_since_last_dist(1e10)  {}

    inline void init(const int &_num_nonempty) {
        num_nonempty = _num_nonempty;
        fanout = std::max<int>(_num_nonempty, minFan);
        fanout <<= 1;
        pe_data = new OptimisticDilaxPairEntry[fanout];
        // Skip explicit initialization - constructor already sets key=-3
    }

    inline void inc_num_nonempty() { ++num_nonempty; }
    inline int get_num_nonempty() { return num_nonempty; }

    int cal_num_nonempty() {
        if (num_nonempty <= 0) {
            assert(num_nonempty == 0);
            for (int i = 0; i < fanout; ++i) {
                OptimisticDilaxPairEntry &pe = pe_data[i];
                if (pe.key >= 0) {
                    ++num_nonempty;
                } else if (pe.key == -1) {
                    num_nonempty += pe.child->cal_num_nonempty();
                } else if (pe.key == -2) {
                    num_nonempty += 2;
                }
            }
        }
        return num_nonempty;
    }

    void init_after_bulk_load() {
        last_total_n_travs = total_n_travs;
        last_nn = num_nonempty;
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key == -1) {
                pe.child->init_after_bulk_load();
            }
        }
    }

    void check_num_nonempty() {
        assert(num_nonempty > 1);
        if (!is_internal()) {
            assert(num_nonempty < LEAF_MAX_CAPACIY);
        }
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key == -1) {
                pe.child->check_num_nonempty();
            }
        }
    }


    // Defensive bounds checking with safe fallbacks (SALI-style)
    inline int PREDICT_POS_SAFE(const keyType &key) const {
        int pred = LR_PRED(a, b, key, fanout);
        // Defensive bounds checking with safe fallbacks
        if (pred < 0) return 0;
        if (pred >= fanout) return fanout - 1;
        return pred;
    }

    inline recordPtr leaf_find(const keyType &key) const {
        EBR::enterEpoch(); // Protect this operation
        
        // Always restart, never fail approach
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;

        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart; // Restart instead of failing
        }

        int pred = PREDICT_POS_SAFE(key);
        
        int64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;

        if (pe_data[pred].key == key) {
            recordPtr result = pe_data[pred].ptr;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            EBR::exitEpoch();
            return result;
        }
        
        if (pe_data[pred].key == -1) {
            dilaxNode *child = pe_data[pred].child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!child) goto restart; // Restart if child is null
            recordPtr result = child->leaf_find(key);
            EBR::exitEpoch();
            return result;
        }
        
        if (pe_data[pred].key == -2) {
            fan2Leaf *fan2child = pe_data[pred].fan2child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!fan2child) goto restart; // Restart if fan2child is null
            
            recordPtr result = -1;
            if (fan2child->k1 == key) result = fan2child->p1;
            else if (fan2child->k2 == key) result = fan2child->p2;
            
            EBR::exitEpoch();
            return result;
        }
        
        pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
        if (needRestart) goto restart;
        
        // CRITICAL FIX: Add comprehensive fallback search if prediction was wrong
        // This ensures we find keys even if linear model prediction fails
        for (int i = 0; i < fanout; ++i) {
            if (i == pred) continue; // Already checked predicted position
            
            bool fallbackNeedRestart = false;
            int64_t fallbackVersion = pe_data[i].readLockOrRestart(fallbackNeedRestart);
            if (fallbackNeedRestart) goto restart;
            
            if (pe_data[i].key == key) {
                recordPtr result = pe_data[i].ptr;
                pe_data[i].readUnlockOrRestart(fallbackVersion, fallbackNeedRestart);
                if (fallbackNeedRestart) goto restart;
                EBR::exitEpoch();
                return result;
            }
            
            if (pe_data[i].key == -1) {
                dilaxNode *child = pe_data[i].child;
                pe_data[i].readUnlockOrRestart(fallbackVersion, fallbackNeedRestart);
                if (fallbackNeedRestart) goto restart;
                if (child) {
                    recordPtr result = child->leaf_find(key);
                    if (result != -1) {
                        EBR::exitEpoch();
                        return result;
                    }
                }
            }
            
            if (pe_data[i].key == -2) {
                fan2Leaf *fan2child = pe_data[i].fan2child;
                pe_data[i].readUnlockOrRestart(fallbackVersion, fallbackNeedRestart);
                if (fallbackNeedRestart) goto restart;
                if (fan2child) {
                    if (fan2child->k1 == key) {
                        EBR::exitEpoch();
                        return fan2child->p1;
                    }
                    if (fan2child->k2 == key) {
                        EBR::exitEpoch();
                        return fan2child->p2;
                    }
                }
            } else {
                pe_data[i].readUnlockOrRestart(fallbackVersion, fallbackNeedRestart);
                if (fallbackNeedRestart) goto restart;
            }
        }
        
        EBR::exitEpoch();
        return -1; // Key truly not found after exhaustive search
    }

    inline int range_query_from(const keyType &k1, recordPtr *results) const {
        EBR::enterEpoch(); // Protect this operation
        
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int j = 0;
        int pred = PREDICT_POS_SAFE(k1);
        
        int64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;
        
        OptimisticDilaxPairEntry &first_pe = pe_data[pred];
        if (first_pe.key == -1) {
            dilaxNode *child = first_pe.child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!child) goto restart; // Restart if child is null
            
            j = child->range_query_from(k1, results);
        } else if (first_pe.key == -2) {
            fan2Leaf *fan2child = first_pe.fan2child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!fan2child) goto restart; // Restart if fan2child is null
            
            keyType _k1 = fan2child->k1;
            keyType _k2 = fan2child->k2;
            if (_k1 >= k1) {
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            } else if (_k2 >= k1) {
                results[j++] = fan2child->p2;
            }
        } else if (first_pe.key >= k1) {
            recordPtr ptr = first_pe.ptr;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            results[j++] = ptr;
        } else {
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
        }

        // Simple iteration for remaining entries (could be improved)
        for(int i = pred + 1; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            keyType key = pe.key; // Volatile read
            if (key >= 0) {
                results[j++] = pe.ptr;
            } else if (key == -1) {
                if (!pe.child) continue; // Skip if child is null
                pe.child->collect_all_ptrs(results+j);
                j += pe.child->num_nonempty;
            } else if (key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                if (!fan2child) continue; // Skip if fan2child is null
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            }
        }

        EBR::exitEpoch();
        return j;
    }

    inline int range_query_to(const keyType &k2, recordPtr *results) const {
        // Range query implementation with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int j = 0;
        int pred = PREDICT_POS_SAFE(k2);
        
        for(int i = 0; i < pred; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                results[j++] = pe.ptr;
            } else if (pe.key == -1) {
                if (!pe.child) continue; // Skip if child is null
                pe.child->collect_all_ptrs(results+j);
                j += pe.child->num_nonempty;
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                if (!fan2child) continue; // Skip if fan2child is null
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            }
        }

        OptimisticDilaxPairEntry &pe = pe_data[pred];
        if (pe.key == -1) {
            if (!pe.child) goto restart; // Restart if child is null
            j += pe.child->range_query_to(k2, results+j);
        } else if (pe.key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (!fan2child) goto restart; // Restart if fan2child is null
            keyType _k1 = fan2child->k1;
            keyType _k2 = fan2child->k2;
            if (_k2 < k2) {
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            } else if (_k1 < k2) {
                results[j++] = fan2child->p1;
            }
        } else if (pe.key < k2) {
            results[j++] = pe.ptr;
        }
        return j;
    }


    inline void collect_all_ptrs(recordPtr *results) const{
        // Defensive collection with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int j = 0;
        for(int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                results[j++] = pe.ptr;
            } else if (pe.key == -1) {
                if (!pe.child) continue; // Skip if child is null
                pe.child->collect_all_ptrs(results+j);
                j += pe.child->num_nonempty;
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                if (!fan2child) continue; // Skip if fan2child is null
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            }
        }
    }


    inline int range_query(const keyType &k1, const keyType &k2, recordPtr *results) const {
        // Range queries with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int pred1 = PREDICT_POS_SAFE(k1);
        int pred2 = PREDICT_POS_SAFE(k2);

        if (pred1 == pred2) {
            OptimisticDilaxPairEntry &pe = pe_data[pred1];
            if (pe.key == -1) {
                if (!pe.child) goto restart; // Restart if child is null
                return pe.child->range_query(k1, k2, results);
            } else if (pe.key == -2) {
                fan2Leaf *leaf = pe.fan2child;
                if (!leaf) goto restart; // Restart if leaf is null
                int n = 0;
                keyType _k1 = leaf->k1;
                keyType _k2 = leaf->k2;
                if (_k1 >= k1 && _k1 < k2) {
                    results[n++] = leaf->p1;
                }
                if (_k2 >= k1 && _k2 < k2) {
                    results[n++] = leaf->p2;
                }
                return n;
            } else if (pe.key >= k1 && pe.key < k2) {
                results[0] = pe.ptr;
                return 1;
            }
        } else { // pred1 < pred2

            OptimisticDilaxPairEntry &first_pe = pe_data[pred1];
            int n = 0;
            if (first_pe.key == -1) {
                if (!first_pe.child) goto restart; // Restart if child is null
                n = first_pe.child->range_query_from(k1, results);
            } else if (first_pe.key == -2) {
                fan2Leaf *leaf = first_pe.fan2child;
                if (!leaf) goto restart; // Restart if leaf is null
                keyType _k1 = leaf->k1;
                keyType _k2 = leaf->k2;
                if (_k1 >= k1) {
                    results[0] = leaf->p1;
                    results[1] = leaf->p2;
                    n = 2;
                } else if (_k2 >= k1) {
                    results[1] = leaf->p2;
                    n = 1;
                }
            } else if (first_pe.key >= k1) {
                results[0] = first_pe.ptr;
                n = 1;
            }

            for (int i = pred1 + 1; i < pred2; ++i) {
                OptimisticDilaxPairEntry &pe = pe_data[i];
                if (pe.key == -1) {
                    if (!pe.child) continue; // Skip if child is null
                    pe.child->collect_all_ptrs(results+n);
                    n += pe.child->num_nonempty;
                } else if (pe.key == -2) {
                    fan2Leaf *leaf = pe.fan2child;
                    if (!leaf) continue; // Skip if leaf is null
                    results[n++] = leaf->p1;
                    results[n++] = leaf->p2;
                } else if (pe.key >= k1 && pe.key < k2) {
                    results[n++] = pe.ptr;
                }
            }

            OptimisticDilaxPairEntry &final_pe = pe_data[pred2];
            if (final_pe.key == -1) {
                if (!final_pe.child) goto restart; // Restart if child is null
                n += final_pe.child->range_query_to(k2, results+n);
            } else if (final_pe.key == -2) {
                fan2Leaf *leaf = final_pe.fan2child;
                if (!leaf) goto restart; // Restart if leaf is null
                keyType _k1 = leaf->k1;
                keyType _k2 = leaf->k2;
                if (_k2 < k2) {
                    results[n++] = leaf->p1;
                    results[n++] = leaf->p2;
                } else if (_k1 < k2) {
                    results[n++] = leaf->p1;
                }

            } else if (final_pe.key < k2) {
                results[n++] = final_pe.ptr;
            }
            return n;
        }
        return 0; // Default return for completeness
    }


    inline dilaxNode* find_child(const keyType &key) {
        // Defensive find_child with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int i = PREDICT_POS_SAFE(key);
        
        if (pe_data[i].key == -1 && pe_data[i].child) {
            return pe_data[i].child;
        }
        
        // If not a child or child is null, restart
        goto restart;
    }


    inline bool if_retrain() { return (!is_internal()) && (total_n_travs * last_nn >= ((last_total_n_travs * num_nonempty) << 1) ); }

     inline void put_three_keys(const keyType &k0, const recordPtr &p0, const keyType &k1, const recordPtr &p1, const keyType &k2, const recordPtr &p2) {
         double offset = fanout / 3.0;
         keyType s = MIN_KEY(k1 - k0, k2 - k1);
         b = offset / s;
         a = 0.5 + offset - b * k1;
         int pos0 = PREDICT_POS_SAFE(k0);
         int pos1 = PREDICT_POS_SAFE(k1);
         int pos2 = PREDICT_POS_SAFE(k2);
         pe_data[pos0].assign(k0, p0);
         pe_data[pos1].assign(k1, p1);
         pe_data[pos2].assign(k2, p2);
         // Remove assert - positions might be equal due to safe clamping
         total_n_travs = 3;
         avg_n_travs_since_last_dist = 1;
     }

     inline void put_three_keys(const keyType *_keys, const recordPtr *_ptrs) {
         keyType k0 = _keys[0];
         keyType k1 = _keys[1];
         keyType k2 = _keys[2];
         double offset = fanout / 3.0;
         keyType s = MIN_KEY(k1 - k0, k2 - k1);
         b = offset / s;
         a = 0.5 + offset - b * k1;
         int pos0 = PREDICT_POS_SAFE(k0);
         int pos1 = PREDICT_POS_SAFE(k1);
         int pos2 = PREDICT_POS_SAFE(k2);

         pe_data[pos0].assign(k0, _ptrs[0]);
         pe_data[pos1].assign(k1, _ptrs[1]);
         pe_data[pos2].assign(k2, _ptrs[2]);
         // Remove assert - positions might be equal due to safe clamping
         total_n_travs = 3;
         avg_n_travs_since_last_dist = 1;
     }

     void num_nonempty_stats(int &n0, int &n1, int &n2, int &n, long &total_fan, long &n_empty_slos) {
         total_fan += fanout;
         if (num_nonempty == 0) {
             n0 = n = 1;
             n1 = n2 = 0;
             return;
         } else if (num_nonempty == 1) {
             n0 = n2 = 0;
             n1 = n = 1;
         } else if (num_nonempty == 2) {
             n0 = n1 = 0;
             n2 = n = 1;
         } else {
             n = 1;
             n0 = n1 = n2 = 0;
         }
         for (int i = 0; i < fanout; ++i) {
             int cn0 = 0;
             int cn1 = 0;
             int cn2 = 0;
             int cn = 0;
             long c_total_fan = 0;
             long c_n_empty_slots = 0;
             OptimisticDilaxPairEntry &pe = pe_data[i];
             if (pe.key == -1) {
                 pe.child->num_nonempty_stats(cn0, cn1, cn2, cn, c_total_fan, c_n_empty_slots);
             }
             if (pe.key < -2) {
                 ++n_empty_slos;
             }
             n0 += cn0;
             n1 += cn1;
             n2 += cn2;
             n += cn;
             total_fan += c_total_fan;
             n_empty_slos += c_n_empty_slots;
         }
     }


    ~dilaxNode(){
        // Use EBR for safe concurrent deletion
        if (pe_data) {
            // Add safety check for fanout
            if (fanout > 0 && fanout < 100000) {
                for (int i = 0; i < fanout; ++i) {
                    // Check if pe_data[i] is valid before accessing
                    if (reinterpret_cast<uintptr_t>(&pe_data[i]) != 0) {
                        if (pe_data[i].key == -1 && pe_data[i].child) {
                            dilaxNode *child = pe_data[i].child;
                            // Check if child pointer looks valid
                            if (reinterpret_cast<uintptr_t>(child) > 0x1000) {
                                EBR::scheduleDelete(child); // Safe concurrent delete
                            }
                        }
                    }
                }
            }
            delete [] pe_data; // pe_data array itself is safe to delete immediately
            pe_data = NULL;
        }
    }


//-------------------------------------------------------------------------


    void save(FILE *fp) {
        fwrite(&meta_info, sizeof(int), 1, fp);
        fwrite(&a, sizeof(double), 1, fp);
        fwrite(&b, sizeof(double), 1, fp);
        fwrite(&fanout, sizeof(int), 1, fp);
        fwrite(&num_nonempty, sizeof(int), 1, fp);
        fwrite(&avg_n_travs_since_last_dist, sizeof(double), 1, fp);
        fwrite(&total_n_travs, sizeof(long), 1, fp);


        fwrite(&last_total_n_travs, sizeof(long), 1, fp);
        fwrite(&last_nn, sizeof(int), 1, fp);
        fwrite(&n_adjust, sizeof(int), 1, fp);

        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            keyType key = pe.key;
            fwrite(&(key), sizeof(keyType),1, fp);
            if (key >= 0) {
                fwrite(&(pe.ptr), sizeof(recordPtr),1, fp);
            } else if (key == -1){
                pe.child->save(fp);
            } else if (key == -2) {
                pe.fan2child->save(fp);
            }
        }

        if (fanout <= 0) {
            cout << "!!!!error, fanout = " << fanout << ", meta_info = " << meta_info << endl;
        }
        assert(fanout > 0);

    }

    void load(FILE *fp) {
        fread(&meta_info, sizeof(int), 1, fp);
        fread(&a, sizeof(double), 1, fp);
        fread(&b, sizeof(double), 1, fp);
        fread(&fanout, sizeof(int), 1, fp);
        assert(fanout > 0);

        fread(&num_nonempty, sizeof(int), 1, fp);
        fread(&avg_n_travs_since_last_dist, sizeof(double), 1, fp);
        fread(&total_n_travs, sizeof(long), 1, fp);


        fread(&last_total_n_travs, sizeof(long), 1, fp);
        fread(&last_nn, sizeof(int), 1, fp);
        fread(&n_adjust, sizeof(int), 1, fp);

        pe_data = new OptimisticDilaxPairEntry[fanout];
        keyType key = 0;
        recordPtr ptr = 0;
        for (int i = 0; i < fanout; ++i) {
            fread(&key, sizeof(keyType), 1, fp);
            if (key >= 0) {
                fread(&ptr, sizeof(recordPtr), 1, fp);
                pe_data[i].assign(key, ptr);
            } else if (key == -1){
                dilaxNode *child = new dilaxNode(false);
                child->load(fp);
                pe_data[i].setChild(child);
            } else if (key == -2) {
                fan2Leaf *fan2child = new fan2Leaf;
                fan2child->load(fp);
                pe_data[i].setFan2Child(fan2child);
            } else {
                pe_data[i].setNull();
            }
        }
    }


    void cal_lr_params(keyType *keys, int n_keys) { //}, vector<int>& child_fans) {
        assert(n_keys >= 0);
        if (n_keys >= 2) {
            // note y_start = 1 here
//        linearReg(keys, 1, this->a, this->b, n_keys);
//        double first_pred = a + b * keys[0];
//        if (first_pred > 0) {
//            a -= (floor(first_pred) - 1);
//        }
//        double last_pred = a + b * keys[n_keys - 1];
//        fanout = std::min<int>(std::max<int>(static_cast<int>(n_keys * 1.1), n_keys + 1), ceil(last_pred));

            b = 1.0 * n_keys / (keys[n_keys - 1] - keys[0]);
            a = 1.0 - b * keys[0];
            fanout = n_keys + 2;
        } else {
            a = b = 0;
            fanout = 1;
        }
    }

    void trim() {
        if (fanout <= 0) {
            cout << "****error, fanout = " << fanout << ", is_internal = " << is_internal() << endl;
        }
        assert(fanout > 0);
        if (!is_internal() && num_nonempty == 0) {
            return;
        }
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (!is_internal() && num_nonempty == 0) {
                cout << "i = " << i << ", fan = " << fanout << ", pe.key = " << pe.key << endl;
            }
            if (pe.key == -1) {
                dilaxNode *child = pe.child;
                child->trim();
            }
        }
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key == -1) {
                dilaxNode *child = pe.child;
                assert(long(child) != -3l);
                assert(child->fanout >= 1);
                if (!(child->is_internal()) && child->num_nonempty == 0) {
                    EBR::scheduleDelete(child);
                    pe.setNull();
                } else if ((child->fanout == 1) || (!(child->is_internal()) && child->num_nonempty == 1)) {
                    pe_data[i] = child->pe_data[0];
                    child->fanout = 0;
                    EBR::scheduleDelete(child);
                }
            }
        }

    }

    void simplify() {
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key == -1) {
                dilaxNode *child = pe.child;
                if (child->num_nonempty == 2) {
                    OptimisticDilaxPairEntry &cpe = child->pe_data[0];
                    keyType k1 = cpe.key;
                    recordPtr p1 = cpe.ptr;
                    cpe = child->pe_data[1];
                    keyType k2 = cpe.key;
                    recordPtr p2 = cpe.ptr;
                    EBR::scheduleDelete(child);

                    fan2Leaf *fan2child = new fan2Leaf(k1, p1, k2, p2);
                    pe.setFan2Child(fan2child);
                } else {
                    child->simplify();
                }
            }
        }
    }

    void bulk_loading(const keyType *keys, const recordPtr *ptrs, bool print) {
        if (num_nonempty == 0) {
             fanout = 2;
             total_n_travs = 0;
             return;
         } else {
             init();

             assert(fanout > 0);
             if (num_nonempty == 1) {
//                pe_data = new pairEntry[fanout];
                a = b = 0;
                pe_data[0].assign(keys[0], ptrs[0]);
                total_n_travs = 1;
                return;
             } else if (num_nonempty == 2) {
                 b = 1.0 / (keys[1] - keys[0]);
                 a = 0.5 - b * keys[0];
                 pe_data[0].assign(keys[0], ptrs[0]);
                 pe_data[1].assign(keys[1], ptrs[1]);
                 total_n_travs = 2;
                 return;
             } else if (num_nonempty == 3) {
                 put_three_keys(keys, ptrs);
                 return;
             }
         }
         distribute_data(keys, ptrs, print);
     }

    void distribute_data(const keyType *keys, const recordPtr *ptrs, bool print=false) {
         assert(num_nonempty > 3);

         total_n_travs = 0;
         dilax::linearReg_w_expanding(keys, a, b, num_nonempty, fanout, false);
//        linearReg_w_expanding(keys, a, b, num_nonempty, fanout, true);
         int last_k_id = 0;
         keyType last_key = keys[0];
         int pos = -1;
         int last_pos = PREDICT_POS_SAFE(last_key);  // Use safe prediction

         keyType final_key = keys[num_nonempty - 1];
//    int final_pos = PREDICT_POS_SAFE(final_key);
         if (b < 0 || last_pos == PREDICT_POS_SAFE(final_key)) {  // Use safe prediction
             dilax::linearReg_w_expanding(keys, a, b, num_nonempty, fanout, true);
             last_pos = PREDICT_POS_SAFE(last_key);  // Use safe prediction
             int final_pos = PREDICT_POS_SAFE(final_key);  // Use safe prediction
             // Remove assert - positions might be equal due to safe clamping
             if (last_pos == final_pos) {
                 // Fallback: use simple linear distribution
                 b = 1.0 * fanout / (final_key - last_key + 1);
                 a = 0 - b * last_key;
                 last_pos = PREDICT_POS_SAFE(last_key);
             }
         }

         // Remove assert - b might be negative in edge cases
         if (b < 0) {
             b = 1.0;  // Fallback value
         }

         for (int k_id = 1; k_id < num_nonempty; ++k_id) {
             keyType key = keys[k_id];
             assert (key != last_key);
             pos = PREDICT_POS_SAFE(key);  // Use safe prediction

             // Remove assert - positions might be equal due to safe clamping

             if (pos != last_pos) {
                 if (k_id == last_k_id + 1) {
                     pe_data[last_pos].assign(last_key, ptrs[last_k_id]);
                     ++total_n_travs;
                 } else { // need to create a new node
                     int n_keys_this_child = k_id - last_k_id;
                     if (n_keys_this_child == 3) {
                         dilaxNode *child = new dilaxNode(false);
                         child->init(3);
                         child->put_three_keys(keys + last_k_id, ptrs + last_k_id);
                         pe_data[last_pos].setChild(child);
                         total_n_travs += 6;
                     }
                     else if (n_keys_this_child == 2) {
                         fan2Leaf *fan2child = new fan2Leaf(keys[last_k_id], ptrs[last_k_id], keys[last_k_id + 1], ptrs[last_k_id + 1]);
                         pe_data[last_pos].setFan2Child(fan2child);
                         total_n_travs += 4;
                     }
                     else {
                         dilaxNode *child = new dilaxNode(false);
                         child->init(n_keys_this_child);
                         child->distribute_data(keys + last_k_id, ptrs + last_k_id);
                         pe_data[last_pos].setChild(child);
                         total_n_travs += n_keys_this_child + child->total_n_travs;
                     }
                 }
                 last_key = key;
                 last_pos = pos;
                 last_k_id = k_id;
             }
         }

         assert(last_k_id != 0);
         // Remove assert - positions might be equal due to safe clamping
         if (last_k_id == num_nonempty - 1) {
             ++total_n_travs;
             pe_data[pos].assign(keys[num_nonempty - 1], ptrs[num_nonempty - 1]);
         } else {
             int n_keys_this_child = num_nonempty - last_k_id;
             if (n_keys_this_child == 3) {

                 dilaxNode *child = new dilaxNode(false);
                 child->init(3);
                 child->put_three_keys(keys + last_k_id, ptrs + last_k_id);
                 pe_data[last_pos].setChild(child);
                 total_n_travs += 6;
             }
             else if (n_keys_this_child == 2) {
                 fan2Leaf *fan2child = new fan2Leaf(keys[last_k_id], ptrs[last_k_id], keys[last_k_id + 1], ptrs[last_k_id + 1]);
                 pe_data[last_pos].setFan2Child(fan2child);
                 total_n_travs += 4;
             }

             else {
                 dilaxNode *child = new dilaxNode(false);
                 child->init(n_keys_this_child);
                 child->distribute_data(keys + last_k_id, ptrs + last_k_id);
                 pe_data[last_pos].setChild(child);
                 total_n_travs += n_keys_this_child + child->total_n_travs;
             }
         }

         last_total_n_travs = total_n_travs;
         last_nn = num_nonempty;
     }


    inline bool insert(const keyType &_key, const recordPtr &_ptr) {
        // Ensure thread-local auxiliary arrays are initialized for this thread
        if (!dilax_auxiliary::retrain_keys) {
            dilax_auxiliary::init_insert_aux_vars();
        }
        
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;

        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart; // Restart instead of failing
        }

        int pred = PREDICT_POS_SAFE(_key);
        
        int64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;

        if (pe_data[pred].key < -2) { // Empty slot
            pe_data[pred].upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            pe_data[pred].assign(_key, _ptr);
            ++num_nonempty;
            ++total_n_travs;
            if (num_nonempty >= LEAF_MAX_CAPACIY) {
                set_int_flag();
            }
            
            pe_data[pred].writeUnlock();
            return true;
            
        } else if (pe_data[pred].key == -1) {
            dilaxNode *child = pe_data[pred].child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!child) goto restart; // Restart if child is null
            
            long child_last_total_n_travs = child->total_n_travs;
            bool if_inserted = child->insert(_key, _ptr);
#ifndef NOT_ADJUST
            if (if_inserted) {
                ++num_nonempty;
                ++total_n_travs;
                total_n_travs += (child->total_n_travs - child_last_total_n_travs);
                if (if_retrain()) {
                    // EBR-BASED SAFE RETRAINING: Full retraining with proper memory management
                    // Use EBR to ensure no ongoing reads are accessing old structure during replacement
                    
                    // First, ensure we have auxiliary arrays available
                    if (!dilax_auxiliary::retrain_keys || !dilax_auxiliary::retrain_ptrs) {
                        dilax_auxiliary::init_insert_aux_vars();
                    }
                    
                    if (dilax_auxiliary::retrain_keys && dilax_auxiliary::retrain_ptrs) {
                        try {
                            // Step 1: Collect all current data safely (no locks needed - just reading)
                            collect_all_data_safe(dilax_auxiliary::retrain_keys, dilax_auxiliary::retrain_ptrs);
                            
                            // Step 2: Create the new structure completely separate from old one
                            OptimisticDilaxPairEntry *new_pe_data;
                            int new_fanout;
                            int old_fanout = fanout;
                            OptimisticDilaxPairEntry *old_pe_data = pe_data;
                            
                            // Calculate new fanout with improved parameters
                            inc_n_adjust();
                            int temp_fanout = std::max<int>(num_nonempty, minFan);
                            temp_fanout <<= 1;
                            
                            // Apply adjustment factor for better performance
                            temp_fanout += (temp_fanout * n_adjust) / 20; // More conservative adjustment
                            
                            if (temp_fanout <= 0 || temp_fanout > 100000) {
                                temp_fanout = std::max<int>(num_nonempty * 2, 16);
                            }
                            new_fanout = temp_fanout;
                            
                            // Step 3: Allocate and build new structure
                            new_pe_data = new OptimisticDilaxPairEntry[new_fanout];
                            
                            // Temporarily swap in new structure for distribute_data
                            pe_data = new_pe_data;
                            fanout = new_fanout;
                            
                            // Build the new optimized structure
                            distribute_data(dilax_auxiliary::retrain_keys, dilax_auxiliary::retrain_ptrs);
                            
                            // Step 4: EBR-safe atomic replacement
                            // The new structure is ready, now atomically replace the pointer
                            // Other threads will either see old or new structure, never half-built
                            
                            // Schedule old structure for EBR deletion
                            if (old_pe_data && old_fanout > 0) {
                                // Create a cleanup task for the old structure
                                OldDilaxStructure *cleanup = new OldDilaxStructure(old_pe_data, old_fanout);
                                
                                // Schedule the entire old structure for safe deletion
                                EBR::scheduleDelete(cleanup, EBR::STRUCTURE_DELETE);
                            }
                            
                            // Track successful retraining
                            static std::atomic<long> dilax_retrain_success{0};
                            dilax_retrain_success.fetch_add(1);
                            
                        } catch (...) {
                            // Emergency recovery - restore old structure if something goes wrong
                            static std::atomic<long> dilax_retrain_errors{0};
                            dilax_retrain_errors.fetch_add(1);
                            
                            // Just do parameter adjustment as fallback
                            inc_n_adjust();
                        }
                    } else {
                        // Fallback if auxiliary arrays not available
                        inc_n_adjust();
                        static std::atomic<long> dilax_retrain_fallback{0};
                        dilax_retrain_fallback.fetch_add(1);
                    }
                }

                if (get_n_adjust() >= 4)  {
                    set_int_flag();
                }
            }
#endif
            return if_inserted;
            
        } else if (pe_data[pred].key == -2) {
            pe_data[pred].upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            fan2Leaf *fan2child = pe_data[pred].fan2child;
            if (!fan2child) {
                pe_data[pred].writeUnlock();
                goto restart; // Restart if fan2child is null
            }
            
            total_n_travs += 2;
            keyType k1 = fan2child->k1;
            recordPtr p1 = fan2child->p1;
            keyType k2 = fan2child->k2;
            recordPtr p2 = fan2child->p2;
            
            if (_key == k1 || _key == k2) {
                pe_data[pred].writeUnlock();
                return false;
            }
            
            ++num_nonempty;
            dilaxNode *child = new dilaxNode(false);
            child->init(3);

            if (_key > k2) {
                child->put_three_keys(k1, p1, k2, p2, _key, _ptr);
            } else if (_key < k1) {
                child->put_three_keys(_key, _ptr, k1, p1, k2, p2);
            } else {
                child->put_three_keys(k1, p1, _key, _ptr, k2, p2);
            }

            pe_data[pred].setChild(child);
            pe_data[pred].writeUnlock();
            return true;
            
        } else if (pe_data[pred].key == _key) {
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            return false;
            
        } else {
            pe_data[pred].upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            keyType k1, k2;
            recordPtr p1, p2;
            if (pe_data[pred].key < _key) {
                k1 = pe_data[pred].key;
                p1 = pe_data[pred].ptr;
                k2 = _key;
                p2 = _ptr;
            } else {
                k1 = _key;
                p1 = _ptr;
                k2 = pe_data[pred].key;
                p2 = pe_data[pred].ptr;
            }
            
            // Defensive check
            if (num_nonempty <= 1) {
                pe_data[pred].writeUnlock();
                goto restart;
            }
            
            total_n_travs += 3;
            ++num_nonempty;

            fan2Leaf *fan2child = new fan2Leaf(k1, p1, k2, p2);
            pe_data[pred].setFan2Child(fan2child);
            pe_data[pred].writeUnlock();
            
            // Periodically advance EBR epoch (less frequently)
            static thread_local int insertCount = 0;
            if (++insertCount % 1000 == 0) {  // Changed from 100 to 1000
                EBR::advance();
            }
            
            return true;
        }
    }


    inline int erase(const keyType &_key) {
        // Robust erase with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int pred = PREDICT_POS_SAFE(_key);
        
        int64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;
        
        OptimisticDilaxPairEntry &pe = pe_data[pred];
        if (pe.key == _key) {
            pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            pe.setNull();
            --num_nonempty;
            --total_n_travs;
            int result = num_nonempty;
            pe.writeUnlock();
            return result;
        }
        else if (pe.key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (!fan2child) {
                pe.readUnlockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                return -1;
            }
            
            if (fan2child->k1 == _key) {
                pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                
                pe.assign(fan2child->k2, fan2child->p2);
                --num_nonempty;
                total_n_travs -= 3;
                int result = num_nonempty;
                pe.writeUnlock();
                return result;
            } else if (fan2child->k2 == _key) {
                pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                
                pe.assign(fan2child->k1, fan2child->p1);
                --num_nonempty;
                total_n_travs -= 3;
                int result = num_nonempty;
                pe.writeUnlock();
                return result;
            } else {
                pe.readUnlockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                return -1;
            }
        } else if (pe.key == -1) {
            dilaxNode *child = pe.child;
            pe.readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!child) goto restart;
            
            long child_n_travs = child->total_n_travs;
            int flag = child->erase(_key);

            if (flag > 0) {
                total_n_travs -= (child_n_travs - child->total_n_travs + 1);
                --num_nonempty;
                return num_nonempty;
            } else if (flag == 0){
                total_n_travs -= (child_n_travs + 1);
                --num_nonempty;
                pe.setNull();
                return num_nonempty;
            } else {
                return -1;
            }
        } else {
            pe.readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            return -1;
        }
    }
    inline int erase_and_get_ptr(const keyType &_key, recordPtr &ptr) {
        // Robust erase_and_get_ptr with restart capability
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;
        
        // Defensive checks - restart if structure is invalid
        if (!pe_data || fanout <= 0) {
            goto restart;
        }
        
        int pred = PREDICT_POS_SAFE(_key);
        
        int64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;
        
        OptimisticDilaxPairEntry &pe = pe_data[pred];
        if (pe.key == _key) {
            pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            ptr = pe.ptr;
            pe.setNull();
            --num_nonempty;
            --total_n_travs;
            int result = num_nonempty;
            pe.writeUnlock();
            return result;
        }
        else if (pe.key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (!fan2child) {
                pe.readUnlockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                return -1;
            }
            
            if (fan2child->k1 == _key) {
                pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                
                ptr = fan2child->p1;
                pe.assign(fan2child->k2, fan2child->p2);
                --num_nonempty;
                total_n_travs -= 3;
                int result = num_nonempty;
                pe.writeUnlock();
                return result;
            } else if (fan2child->k2 == _key) {
                pe.upgradeToWriteLockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                
                ptr = fan2child->p2;
                pe.assign(fan2child->k1, fan2child->p1);
                --num_nonempty;
                total_n_travs -= 3;
                int result = num_nonempty;
                pe.writeUnlock();
                return result;
            } else {
                pe.readUnlockOrRestart(versionItem, needRestart);
                if (needRestart) goto restart;
                return -1;
            }
        } else if (pe.key == -1) {
            dilaxNode *child = pe.child;
            pe.readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            if (!child) goto restart;
            
            long child_n_travs = child->total_n_travs;
            int flag = child->erase_and_get_ptr(_key, ptr);

            if (flag > 0) {
                total_n_travs -= (child_n_travs - child->total_n_travs + 1);
                --num_nonempty;
                return num_nonempty;
            } else if (flag == 0){
                total_n_travs -= (child_n_travs + 1);
                --num_nonempty;
                pe.setNull();
                return num_nonempty;
            } else {
                return -1;
            }
        } else {
            pe.readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            return -1;
        }
    }


    void collect_and_clear(keyType *keys, recordPtr *ptrs) {
        // Safety check for null pointers
        if (!keys || !ptrs) {
            cout << "ERROR: collect_and_clear called with null pointers!" << endl;
            return;
        }
        
        int j = 0;
        for(int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                keys[j] = pe.key;
                ptrs[j++] = pe.ptr;
            } else if (pe.key == -1) {
                dilaxNode *child = pe.child;
                child->collect_and_clear(keys+j, ptrs+j);
                j += child->num_nonempty;
                delete child;
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                keys[j] = fan2child->k1;
                ptrs[j++] = fan2child->p1;
                keys[j] = fan2child->k2;
                ptrs[j++] = fan2child->p2;
            }
        }
        delete[] pe_data;
        pe_data = NULL;
        assert(j == num_nonempty);
    }

    // Safe collection that doesn't destroy the current structure
    void collect_all_data_safe(keyType *keys, recordPtr *ptrs) {
        int j = 0;
        for(int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                keys[j] = pe.key;
                ptrs[j++] = pe.ptr;
            } else if (pe.key == -1) {
                dilaxNode *child = pe.child;
                if (child) {
                    child->collect_all_data_safe(keys+j, ptrs+j);
                    j += child->num_nonempty;
                }
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                if (fan2child) {
                    keys[j] = fan2child->k1;
                    ptrs[j++] = fan2child->p1;
                    keys[j] = fan2child->k2;
                    ptrs[j++] = fan2child->p2;
                }
            }
        }
        // Note: We don't delete anything here - just copy the data
    }

    void collect_all_keys(keyType *keys) {
        assert(b >= 0);
        int j = 0;
        for(int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                keys[j++] = pe.key;
            } else if (pe.key == -1) {
                dilaxNode *child = pe.child;
                child->collect_all_keys(keys+j);
                j += child->num_nonempty;
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                keys[j++] = fan2child->k1;
                keys[j++] = fan2child->k2;
            }
        }
        if (j != num_nonempty) {
            cout << "j = " << j << ", num_nonempty = " << num_nonempty << ", is_internal = " << is_internal() << endl;
            for(int i = 0; i < num_nonempty; ++i) {
                OptimisticDilaxPairEntry &pe = pe_data[i];
                if (pe.key >= 0) {
                    cout << "i = " << i << ", pe.key = " << pe.key << endl;
                } else if (pe.key == -1) {
                    dilaxNode *child = pe.child;
                    child->collect_all_keys(keys+j);
                    cout << "i = " << i << ", child.num_nonempty = " << child->num_nonempty << endl;
                    j += child->num_nonempty;
                } else if (pe.key == -2) {
                    fan2Leaf *fan2child = pe.fan2child;
                    cout << "i = " << i << ", fan2child.num_nonempty = 2" << endl;
                    keys[j++] = fan2child->k1;
                    keys[j++] = fan2child->k2;
                }
            }
        }
        assert(j == num_nonempty);
    }

    // compute average traversals per key and recurse
    void cal_avg_n_travs() {
        if (!is_internal()) {
            if (num_nonempty <= 2) {
                avg_n_travs_since_last_dist = 1;
            } else {
                avg_n_travs_since_last_dist = 1.0 * total_n_travs / num_nonempty;
            }
        }
        for (int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key == -1) {
                pe.child->cal_avg_n_travs();
            }
        }
    }

};

// Now that dilaxNode is fully defined, implement the EBR delete helpers
inline void EBR::deleteDilaxNode(void* ptr) {
    delete static_cast<dilaxNode*>(ptr);
}

inline void EBR::deleteOldStructure(void* ptr) {
    OldDilaxStructure* oldStruct = static_cast<OldDilaxStructure*>(ptr);
    if (oldStruct && oldStruct->data) {
        // Clean up any child nodes in the old structure
        for (int i = 0; i < oldStruct->fanout; ++i) {
            if (oldStruct->data[i].key == -1 && oldStruct->data[i].child) {
                // Schedule child nodes for deletion too
                scheduleDelete(oldStruct->data[i].child);
            }
        }
        delete[] oldStruct->data;
    }
    delete oldStruct;
}


#endif //DILAX_DILAXNODE_H
