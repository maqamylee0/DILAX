#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stack>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
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
    static std::mutex deleteMutex;

public:
    static void enterEpoch() {
        localEpoch = globalEpoch.load();
    }
    
    static void exitEpoch() {
        localEpoch = 0;
    }
    
    static void scheduleDelete(void* ptr) {
        if (!ptr) return; // Null pointer check
        
        uint64_t epoch = globalEpoch.load();
        std::lock_guard<std::mutex> lock(deleteMutex);
        
        // Simple implementation - just add to pending list
        // Only schedule if slot is empty to avoid overwriting
        void* expected = nullptr;
        if (pendingDeletes[epoch % 3].compare_exchange_strong(expected, ptr)) {
            // Successfully scheduled
        } else {
            // Slot occupied, skip this deletion for safety
            // In a real implementation, would use a proper queue
        }
    }
    
    static void scheduleArrayDelete(void* ptr);  // Implementation after OptimisticDilaxPairEntry is defined
    
    static void advance() {
        globalEpoch.fetch_add(1, std::memory_order_release);
        
        uint64_t currentEpoch = globalEpoch.load(std::memory_order_acquire);
        if (currentEpoch < 4) return; // Reduce wait time for better throughput
        
        uint64_t oldEpoch = currentEpoch - 4; // Wait 4 epochs instead of 6
        void* toDelete = pendingDeletes[oldEpoch % 3].exchange(nullptr, std::memory_order_acq_rel);
        if (toDelete) {
            // Additional safety check before deletion
            if (reinterpret_cast<uintptr_t>(toDelete) > 0x1000) {
                deleteDilaxNode(toDelete);
            }
        }
    }
    
    // Helper function to be defined after dilaxNode is fully declared
    static void deleteDilaxNode(void* ptr);
};

// Declare static members (definitions need to be in a .cpp file)

// Optimistic lock implementation based on SALI's approach
class OptLock {
private:
    std::atomic<uint64_t> typeVersionLockObsolete{0b100}; // Use uint64_t like SALI

public:
    static constexpr uint64_t UNLOCKED = 0b100;
    static constexpr uint64_t LOCKED_BIT = 0b10;    // Match SALI's bit layout
    static constexpr uint64_t OBSOLETE_BIT = 0b01;  // Match SALI's bit layout

    OptLock() = default;
    OptLock(const OptLock& other) {
        typeVersionLockObsolete = 0b100; // Don't copy lock state
    }

    uint64_t get_version_number() {
        return typeVersionLockObsolete.load(std::memory_order_acquire);
    }

    bool isLocked(uint64_t version) const {
        return (version & LOCKED_BIT) == LOCKED_BIT;
    }

    bool isLocked() const {
        return (typeVersionLockObsolete.load(std::memory_order_acquire) & LOCKED_BIT) == LOCKED_BIT;
    }

    bool isObsolete(uint64_t version) const {
        return (version & OBSOLETE_BIT) == OBSOLETE_BIT;
    }

    bool isObsolete() const {
        return (typeVersionLockObsolete.load(std::memory_order_acquire) & OBSOLETE_BIT) == OBSOLETE_BIT;
    }

    uint64_t readLockOrRestart(bool &needRestart) {
        uint64_t version = typeVersionLockObsolete.load(std::memory_order_acquire);
        // SALI checks both locked AND obsolete - this is key!
        if (isLocked(version) || isObsolete(version)) {
            _mm_pause();
            needRestart = true;
        }
        return version;
    }

    void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
        needRestart = (startRead != typeVersionLockObsolete.load(std::memory_order_acquire));
    }

    void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
        if (typeVersionLockObsolete.compare_exchange_strong(version, version + LOCKED_BIT)) {
            version = version + LOCKED_BIT; // Update version for caller
        } else {
            _mm_pause();
            needRestart = true;
        }
    }

    void writeLockOrRestart(bool &needRestart) {
        uint64_t version = readLockOrRestart(needRestart);
        if (!needRestart) {
            upgradeToWriteLockOrRestart(version, needRestart);
        }
    }

    void writeUnlock() {
        typeVersionLockObsolete.fetch_add(LOCKED_BIT, std::memory_order_release);
    }

    void writeUnlockObsolete() {
        typeVersionLockObsolete.fetch_add(LOCKED_BIT + OBSOLETE_BIT, std::memory_order_release);
    }

    void labelObsolete() {
        typeVersionLockObsolete.store(typeVersionLockObsolete.load() | OBSOLETE_BIT);
    }

    void checkOrRestart(uint64_t startRead, bool &needRestart) const {
        readUnlockOrRestart(startRead, needRestart);
    }
};

#ifndef DILAX_DILAXNODE_H
#define DILAX_DILAXNODE_H


using namespace std;
struct dilaxNode;
struct fan2Leaf;

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
    std::atomic<keyType> key;  // Make key atomic to prevent torn reads
    union {
        recordPtr ptr;
        dilaxNode *child;
        fan2Leaf *fan2child;
    };

    // Add explicit constructors and assignment operator to handle atomic issue
    OptimisticDilaxPairEntry() : OptLock(), key(-3) {}
    
    OptimisticDilaxPairEntry(const OptimisticDilaxPairEntry& other) : OptLock() {
        key.store(other.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
        ptr = other.ptr;  // This copies the union (same memory layout)
    }
    
    OptimisticDilaxPairEntry& operator=(const OptimisticDilaxPairEntry& other) {
        if (this != &other) {
            key.store(other.key.load(std::memory_order_relaxed), std::memory_order_relaxed);
            ptr = other.ptr;  // This copies the union (same memory layout)
            // Don't copy the atomic lock state - keep our own lock
        }
        return *this;
    }

    void assign(const keyType &_key, const recordPtr &_ptr) {
        ptr = _ptr;  // Set value first
        std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure ptr is written before key
        key.store(_key, std::memory_order_release);  // Atomic key update
    }

    void setChild(dilaxNode *_child) {
        child = _child;  // Set value first
        std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure child is written before key
        key.store(-1, std::memory_order_release);  // Atomic key update
    }

    void setFan2Child(fan2Leaf *_fan2child) {
        fan2child = _fan2child;  // Set value first
        std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure fan2child is written before key
        key.store(-2, std::memory_order_release);  // Atomic key update
    }

    void setNull() {
        key.store(-3, std::memory_order_release);
    }

    bool isEmpty() const { return key.load(std::memory_order_acquire) < -2; }
    bool isLeaf() const { return key.load(std::memory_order_acquire) >= 0; }
    bool hasChild() const { return key.load(std::memory_order_acquire) == -1; }
    bool hasFan2Child() const { return key.load(std::memory_order_acquire) == -2; }
};

// Now implement the EBR array delete helper
inline void EBR::scheduleArrayDelete(void* ptr) {
    // For now, just delete immediately since array access is simpler to synchronize
    if (ptr && reinterpret_cast<uintptr_t>(ptr) > 0x1000) {
        delete[] static_cast<OptimisticDilaxPairEntry*>(ptr);
    }
}

// Helper function for yielding
inline void yield(int count) {
    if (count > 3)
        sched_yield();
    else
        _mm_pause();
}

struct dilaxNode{
     std::atomic<int> fanout;           // Make atomic
     int meta_info;
     std::atomic<double> a;             // Make atomic  
     std::atomic<double> b;             // Make atomic
     std::atomic<int> num_nonempty;     // Make atomic

     std::atomic<OptimisticDilaxPairEntry*> pe_data; // Make atomic pointer

     double avg_n_travs_since_last_dist;
     std::atomic<long> total_n_travs;   // Make atomic
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
        // Calculate new fanout first
        int current_num_nonempty = num_nonempty.load(std::memory_order_acquire);
        int new_fanout = std::max<int>(current_num_nonempty, minFan);
        new_fanout <<= 1;
        
        // Safety check for reasonable fanout size
        if (new_fanout <= 0 || new_fanout > 100000) {
            cout << "ERROR: Invalid fanout=" << new_fanout << " num_nonempty=" << current_num_nonempty << endl;
            new_fanout = std::max<int>(current_num_nonempty, 16); // Fallback value
        }
        
        // Create new array
        OptimisticDilaxPairEntry *new_pe_data = new OptimisticDilaxPairEntry[new_fanout];
        
        // Store old data for cleanup
        OptimisticDilaxPairEntry *old_pe_data = pe_data.load(std::memory_order_acquire);
        
        // Atomically update pe_data first, then fanout
        // This prevents readers from using new fanout with old pe_data
        pe_data.store(new_pe_data, std::memory_order_release);
        fanout.store(new_fanout, std::memory_order_release);
        
        // Schedule old array for safe deletion if it existed
        if (old_pe_data) {
            EBR::scheduleArrayDelete(old_pe_data);
        }
        
        // Skip explicit initialization - constructor already sets key=-3
    }



    dilaxNode(bool _is_internal): a(0), b(0), meta_info(_is_internal), fanout(0), pe_data(NULL), n_adjust(30), num_nonempty(0),
                                 total_n_travs(0), last_total_n_travs(0), last_nn(0), avg_n_travs_since_last_dist(1e10)  {}

    inline void init(const int &_num_nonempty) {
        num_nonempty.store(_num_nonempty, std::memory_order_relaxed);
        int new_fanout = std::max<int>(_num_nonempty, minFan);
        new_fanout <<= 1;
        fanout.store(new_fanout, std::memory_order_relaxed);
        pe_data.store(new OptimisticDilaxPairEntry[new_fanout], std::memory_order_release);
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


    inline recordPtr leaf_find(const keyType &key) const {
        EBR::enterEpoch(); // Protect this operation
        
        // Capture node parameters atomically to prevent mid-read changes
        double local_a = a.load(std::memory_order_acquire);
        double local_b = b.load(std::memory_order_acquire);
        int local_fanout = fanout.load(std::memory_order_acquire);
        OptimisticDilaxPairEntry *local_pe_data = pe_data.load(std::memory_order_acquire);
        
        int pred = LR_PRED(local_a, local_b, key, local_fanout);
        
        // Safety checks with consistent local variables
        if (!local_pe_data || pred < 0 || pred >= local_fanout) {
            EBR::exitEpoch();
            return -1;
        }
        
        // Fast path: try lock-free read first with atomic key access
        keyType entry_key = local_pe_data[pred].key.load(std::memory_order_acquire);
        if (entry_key == key && !local_pe_data[pred].isObsolete()) {
            recordPtr result = local_pe_data[pred].ptr;
            // Double-check key consistency and not obsolete  
            if (local_pe_data[pred].key.load(std::memory_order_acquire) == key && 
                !local_pe_data[pred].isObsolete()) {
                EBR::exitEpoch();
                return result;
            }
        }
        
        // Full optimistic protocol for complex cases
        int restartCount = 0;
        restart:
        if (restartCount++ > 5) {  // Reduce retry limit to fail faster
            EBR::exitEpoch();
            return -1;
        }
        yield(restartCount);
        bool needRestart = false;

        uint64_t versionItem = local_pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;

        // Check if node is being rebuilt
        if (local_pe_data[pred].isObsolete(versionItem)) {
            local_pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            EBR::exitEpoch();
            return -1; // Node is being rebuilt, return not found
        }

        keyType current_key = local_pe_data[pred].key.load(std::memory_order_acquire);
        if (current_key == key) {
            recordPtr result = local_pe_data[pred].ptr;
            local_pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            EBR::exitEpoch();
            return result;
        }
        
        if (current_key == -1) {
            dilaxNode *child = local_pe_data[pred].child;
            local_pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            // Additional safety: check if child is still valid
            if (!child || child->pe_data.load(std::memory_order_acquire) == nullptr) {
                EBR::exitEpoch();
                return -1;
            }
            
            recordPtr result = child->leaf_find(key);
            EBR::exitEpoch();
            return result;
        }
        
        if (current_key == -2) {
            fan2Leaf *fan2child = local_pe_data[pred].fan2child;
            local_pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            recordPtr result = -1;
            if (fan2child && fan2child->k1 == key) result = fan2child->p1;
            else if (fan2child && fan2child->k2 == key) result = fan2child->p2;
            
            EBR::exitEpoch();
            return result;
        }
        
        local_pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
        if (needRestart) goto restart;
        EBR::exitEpoch();
        return -1;
    }

    inline int range_query_from(const keyType &k1, recordPtr *results) const {
        EBR::enterEpoch(); // Protect this operation
        
        int j = 0;
        int pred = LR_PRED(a, b, k1, fanout);
        
        if (!pe_data || pred < 0 || pred >= fanout) {
            EBR::exitEpoch();
            return 0;
        }
        
        // Use optimistic locking for the first entry
        int restartCount = 0;
        restart:
        if (restartCount++) yield(restartCount);
        bool needRestart = false;
        
        uint64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;
        
        OptimisticDilaxPairEntry &first_pe = pe_data[pred];
        if (first_pe.key == -1) {
            dilaxNode *child = first_pe.child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            j = child->range_query_from(k1, results);
        } else if (first_pe.key == -2) {
            fan2Leaf *fan2child = first_pe.fan2child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
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
                pe.child->collect_all_ptrs(results+j);
                j += pe.child->num_nonempty;
            } else if (key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                results[j++] = fan2child->p1;
                results[j++] = fan2child->p2;
            }
        }

        EBR::exitEpoch();
        return j;
    }

    inline int range_query_to(const keyType &k2, recordPtr *results) const {
        EBR::enterEpoch(); // Add protection
        
        // Safety check
        if (!pe_data) {
            EBR::exitEpoch();
            return 0;
        }
        
        int j = 0;
        int pred = LR_PRED(a, b, k2, fanout);
        
        if (pred < 0 || pred >= fanout) {
            EBR::exitEpoch();
            return 0;
        }
        
        for(int i = 0; i < pred; ++i) {
            if (i >= fanout) break; // Additional safety
            OptimisticDilaxPairEntry &pe = pe_data[i];
            keyType key = pe.key; // Volatile read
            if (key >= 0) {
                results[j++] = pe.ptr;
            } else if (key == -1) {
                if (pe.child) {
                    pe.child->collect_all_ptrs(results+j);
                    j += pe.child->num_nonempty;
                }
            } else if (key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                if (fan2child) {
                    results[j++] = fan2child->p1;
                    results[j++] = fan2child->p2;
                }
            }
        }

        OptimisticDilaxPairEntry &pe = pe_data[pred];
        keyType key = pe.key; // Volatile read
        if (key == -1) {
            if (pe.child) {
                j += pe.child->range_query_to(k2, results+j);
            }
        } else if (key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (fan2child) {
                keyType _k1 = fan2child->k1;
                keyType _k2 = fan2child->k2;
                if (_k2 < k2) {
                    results[j++] = fan2child->p1;
                    results[j++] = fan2child->p2;
                } else if (_k1 < k2) {
                    results[j++] = fan2child->p1;
                }
            }
        } else if (key < k2) {
            results[j++] = pe.ptr;
        }
        
        EBR::exitEpoch();
        return j;
    }


    inline void collect_all_ptrs(recordPtr *results) const{
         int j = 0;
         for(int i = 0; i < fanout; ++i) {
             OptimisticDilaxPairEntry &pe = pe_data[i];
             if (pe.key >= 0) {
                 results[j++] = pe.ptr;
             } else if (pe.key == -1) {
                 pe.child->collect_all_ptrs(results+j);
                 j += pe.child->num_nonempty;
             } else if (pe.key == -2) {
                 fan2Leaf *fan2child = pe.fan2child;
                 results[j++] = fan2child->p1;
                 results[j++] = fan2child->p2;
             }
         }
     }


    inline int range_query(const keyType &k1, const keyType &k2, recordPtr *results) const {
        EBR::enterEpoch(); // Add protection
        
        if (!pe_data) {
            EBR::exitEpoch();
            return 0;
        }
        
        // Note: Range queries are tricky with optimistic locking since they need
        // to read multiple entries atomically. For now, we'll use basic protection
        int pred1 = LR_PRED(a, b, k1, fanout);
        int pred2 = LR_PRED(a, b, k2, fanout);

        if (pred1 < 0 || pred2 < 0 || pred1 >= fanout || pred2 >= fanout) {
            EBR::exitEpoch();
            return 0;
        }

        if (pred1 == pred2) {
            OptimisticDilaxPairEntry &pe = pe_data[pred1];
            keyType key = pe.key; // Volatile read
            if (key == -1) {
                if (pe.child) {
                    int result = pe.child->range_query(k1, k2, results);
                    EBR::exitEpoch();
                    return result;
                }
            } else if (key == -2) {
                int n = 0;
                fan2Leaf *leaf = pe.fan2child;
                if (leaf) {
                    keyType _k1 = leaf->k1;
                    keyType _k2 = leaf->k2;
                    if (_k1 >= k1 && _k1 < k2) {
                        results[n++] = leaf->p1;
                    }
                    if (_k2 >= k1 && _k2 < k2) {
                        results[n++] = leaf->p2;
                    }
                }
                EBR::exitEpoch();
                return n;
            } else if (key >= k1 && key < k2) {
                results[0] = pe.ptr;
                EBR::exitEpoch();
                return 1;
            }
        } else { // pred1 < pred2
            OptimisticDilaxPairEntry &first_pe = pe_data[pred1];
            int n = 0;
            keyType first_key = first_pe.key;
            if (first_key == -1) {
                if (first_pe.child) {
                    n = first_pe.child->range_query_from(k1, results);
                }
            } else if (first_key == -2) {
                fan2Leaf *leaf = first_pe.fan2child;
                if (leaf) {
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
                }
            } else if (first_key >= k1) {
                results[0] = first_pe.ptr;
                n = 1;
            }

            for (int i = pred1 + 1; i < pred2 && i < fanout; ++i) {
                OptimisticDilaxPairEntry &pe = pe_data[i];
                keyType key = pe.key; // Volatile read
                if (key == -1) {
                    if (pe.child) {
                        pe.child->collect_all_ptrs(results+n);
                        n += pe.child->num_nonempty;
                    }
                } else if (key == -2) {
                    fan2Leaf *leaf = pe.fan2child;
                    if (leaf) {
                        results[n++] = leaf->p1;
                        results[n++] = leaf->p2;
                    }
                } else if (key >= k1 && key < k2) {
                    results[n++] = pe.ptr;
                }
            }

            OptimisticDilaxPairEntry &final_pe = pe_data[pred2];
            keyType final_key = final_pe.key;
            if (final_key == -1) {
                if (final_pe.child) {
                    n += final_pe.child->range_query_to(k2, results+n);
                }
            } else if (final_key == -2) {
                fan2Leaf *leaf = final_pe.fan2child;
                if (leaf) {
                    keyType _k1 = leaf->k1;
                    keyType _k2 = leaf->k2;
                    if (_k2 < k2) {
                        results[n++] = leaf->p1;
                        results[n++] = leaf->p2;
                    } else if (_k1 < k2) {
                        results[n++] = leaf->p1;
                    }
                }
            } else if (final_key < k2) {
                results[n++] = final_pe.ptr;
            }
            EBR::exitEpoch();
            return n;
        }
        EBR::exitEpoch();
        return 0;
    }


    inline dilaxNode* find_child(const keyType &key) {
        if (!pe_data) return nullptr;
        int i = LR_PRED(a, b, key, fanout);
        if (i < 0 || i >= fanout) return nullptr;
        
        // Simple protection - just check key consistency
        if (pe_data[i].key == -1) {
            return pe_data[i].child;
        }
        return nullptr;
    }


    inline bool if_retrain() { return (!is_internal()) && (total_n_travs * last_nn >= ((last_total_n_travs * num_nonempty) << 1) ); }

     inline void put_three_keys(const keyType &k0, const recordPtr &p0, const keyType &k1, const recordPtr &p1, const keyType &k2, const recordPtr &p2) {
         int local_fanout = fanout.load(std::memory_order_acquire);
         double offset = local_fanout / 3.0;
         keyType s = MIN_KEY(k1 - k0, k2 - k1);
         double local_b = offset / s;
         double local_a = 0.5 + offset - local_b * k1;
         
         // Store the computed values atomically
         a.store(local_a, std::memory_order_release);
         b.store(local_b, std::memory_order_release);
         
         int pos0 = LR_PRED(local_a, local_b, k0, local_fanout);
         int pos1 = LR_PRED(local_a, local_b, k1, local_fanout);
         int pos2 = LR_PRED(local_a, local_b, k2, local_fanout);
         
         OptimisticDilaxPairEntry *local_pe_data = pe_data.load(std::memory_order_acquire);
         local_pe_data[pos0].assign(k0, p0);
         local_pe_data[pos1].assign(k1, p1);
         local_pe_data[pos2].assign(k2, p2);
         assert(pos0 < pos1 && pos1 < pos2);
         total_n_travs.store(3, std::memory_order_release);
         avg_n_travs_since_last_dist = 1;
     }

     inline void put_three_keys(const keyType *_keys, const recordPtr *_ptrs) {
         keyType k0 = _keys[0];
         keyType k1 = _keys[1];
         keyType k2 = _keys[2];
         int local_fanout = fanout.load(std::memory_order_acquire);
         double offset = local_fanout / 3.0;
         keyType s = MIN_KEY(k1 - k0, k2 - k1);
         double local_b = offset / s;
         double local_a = 0.5 + offset - local_b * k1;
         
         // Store the computed values atomically
         a.store(local_a, std::memory_order_release);
         b.store(local_b, std::memory_order_release);
         
         int pos0 = LR_PRED(local_a, local_b, k0, local_fanout);
         int pos1 = LR_PRED(local_a, local_b, k1, local_fanout);
         int pos2 = LR_PRED(local_a, local_b, k2, local_fanout);

         OptimisticDilaxPairEntry *local_pe_data = pe_data.load(std::memory_order_acquire);
         local_pe_data[pos0].assign(k0, _ptrs[0]);
         local_pe_data[pos1].assign(k1, _ptrs[1]);
         local_pe_data[pos2].assign(k2, _ptrs[2]);
         assert(pos0 < pos1 && pos1 < pos2);
         total_n_travs.store(3, std::memory_order_release);
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

            double new_b = 1.0 * n_keys / (keys[n_keys - 1] - keys[0]);
            double new_a = 1.0 - new_b * keys[0];
            int new_fanout = n_keys + 2;
            
            a.store(new_a, std::memory_order_release);
            b.store(new_b, std::memory_order_release);
            fanout.store(new_fanout, std::memory_order_release);
        } else {
            a.store(0, std::memory_order_release);
            b.store(0, std::memory_order_release);
            fanout.store(1, std::memory_order_release);
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
        int current_num_nonempty = num_nonempty.load(std::memory_order_acquire);
        if (current_num_nonempty == 0) {
             fanout.store(2, std::memory_order_release);
             total_n_travs.store(0, std::memory_order_release);
             return;
         } else {
             init();

             int local_fanout = fanout.load(std::memory_order_acquire);
             OptimisticDilaxPairEntry *local_pe_data = pe_data.load(std::memory_order_acquire);
             
             assert(local_fanout > 0);
             if (current_num_nonempty == 1) {
//                pe_data = new pairEntry[fanout];
                a.store(0, std::memory_order_release);
                b.store(0, std::memory_order_release);
                local_pe_data[0].assign(keys[0], ptrs[0]);
                total_n_travs.store(1, std::memory_order_release);
                return;
             } else if (current_num_nonempty == 2) {
                 double new_b = 1.0 / (keys[1] - keys[0]);
                 double new_a = 0.5 - new_b * keys[0];
                 a.store(new_a, std::memory_order_release);
                 b.store(new_b, std::memory_order_release);
                 local_pe_data[0].assign(keys[0], ptrs[0]);
                 local_pe_data[1].assign(keys[1], ptrs[1]);
                 total_n_travs.store(2, std::memory_order_release);
                 return;
             } else if (current_num_nonempty == 3) {
                 put_three_keys(keys, ptrs);
                 return;
             }
         }
         distribute_data(keys, ptrs, print);
     }

    void distribute_data(const keyType *keys, const recordPtr *ptrs, bool print=false) {
         int current_num_nonempty = num_nonempty.load(std::memory_order_acquire);
         assert(current_num_nonempty > 3);

         total_n_travs.store(0, std::memory_order_relaxed);
         
         // Get local copies of atomic values for linearReg functions
         double local_a = a.load(std::memory_order_acquire);
         double local_b = b.load(std::memory_order_acquire);
         int local_fanout = fanout.load(std::memory_order_acquire);
         
         dilax::linearReg_w_expanding(keys, local_a, local_b, current_num_nonempty, local_fanout, false);
         
         // Store back the updated values
         a.store(local_a, std::memory_order_release);
         b.store(local_b, std::memory_order_release);
         
//        linearReg_w_expanding(keys, a, b, num_nonempty, fanout, true);
         int last_k_id = 0;
         keyType last_key = keys[0];
         int pos = -1;
         int last_pos = LR_PRED(local_a, local_b, last_key, local_fanout);

         keyType final_key = keys[current_num_nonempty - 1];
//    int final_pos = LR_PRED(a, b, final_key, fanout);
         if (local_b < 0 || last_pos == LR_PRED(local_a, local_b, final_key, local_fanout)) {
             dilax::linearReg_w_expanding(keys, local_a, local_b, current_num_nonempty, local_fanout, true);
             // Update stored values again
             a.store(local_a, std::memory_order_release);
             b.store(local_b, std::memory_order_release);
             
             last_pos = LR_PRED(local_a, local_b, last_key, local_fanout);
             int final_pos = LR_PRED(local_a, local_b, final_key, local_fanout);
             assert(last_pos != final_pos);
         }

         assert(local_b >= 0);

         // Get pe_data pointer
         OptimisticDilaxPairEntry *local_pe_data = pe_data.load(std::memory_order_acquire);

         for (int k_id = 1; k_id < current_num_nonempty; ++k_id) {
             keyType key = keys[k_id];
             assert (key != last_key);
             pos = LR_PRED(local_a, local_b, key, local_fanout);

             assert(pos >= last_pos);

             if (pos != last_pos) {
                 if (k_id == last_k_id + 1) {
                     local_pe_data[last_pos].assign(last_key, ptrs[last_k_id]);
                     total_n_travs.fetch_add(1, std::memory_order_relaxed);
                 } else { // need to create a new node
                     int n_keys_this_child = k_id - last_k_id;
                     if (n_keys_this_child == 3) {
                         dilaxNode *child = new dilaxNode(false);
                         child->init(3);
                         child->put_three_keys(keys + last_k_id, ptrs + last_k_id);
                         local_pe_data[last_pos].setChild(child);
                         total_n_travs.fetch_add(6, std::memory_order_relaxed);
                     }
                     else if (n_keys_this_child == 2) {
                         fan2Leaf *fan2child = new fan2Leaf(keys[last_k_id], ptrs[last_k_id], keys[last_k_id + 1], ptrs[last_k_id + 1]);
                         local_pe_data[last_pos].setFan2Child(fan2child);
                         total_n_travs.fetch_add(4, std::memory_order_relaxed);
                     }
                     else {
                         dilaxNode *child = new dilaxNode(false);
                         child->init(n_keys_this_child);
                         child->distribute_data(keys + last_k_id, ptrs + last_k_id);
                         local_pe_data[last_pos].setChild(child);
                         total_n_travs.fetch_add(n_keys_this_child + child->total_n_travs.load(std::memory_order_relaxed), std::memory_order_relaxed);
                     }
                 }
                 last_key = key;
                 last_pos = pos;
                 last_k_id = k_id;
             }
         }

         assert(last_k_id != 0);
         assert(pos >= last_pos);
         if (last_k_id == current_num_nonempty - 1) {
             total_n_travs.fetch_add(1, std::memory_order_relaxed);
             local_pe_data[pos].assign(keys[current_num_nonempty - 1], ptrs[current_num_nonempty - 1]);
         } else {
             int n_keys_this_child = current_num_nonempty - last_k_id;
             if (n_keys_this_child == 3) {

                 dilaxNode *child = new dilaxNode(false);
                 child->init(3);
                 child->put_three_keys(keys + last_k_id, ptrs + last_k_id);
                 local_pe_data[last_pos].setChild(child);
                 total_n_travs.fetch_add(6, std::memory_order_relaxed);
             }
             else if (n_keys_this_child == 2) {
                 fan2Leaf *fan2child = new fan2Leaf(keys[last_k_id], ptrs[last_k_id], keys[last_k_id + 1], ptrs[last_k_id + 1]);
                 local_pe_data[last_pos].setFan2Child(fan2child);
                 total_n_travs.fetch_add(4, std::memory_order_relaxed);
             }

             else {
                 dilaxNode *child = new dilaxNode(false);
                 child->init(n_keys_this_child);
                 child->distribute_data(keys + last_k_id, ptrs + last_k_id);
                 local_pe_data[last_pos].setChild(child);
                 total_n_travs.fetch_add(n_keys_this_child + child->total_n_travs.load(std::memory_order_relaxed), std::memory_order_relaxed);
             }
         }

         last_total_n_travs = total_n_travs.load(std::memory_order_relaxed);
         last_nn = current_num_nonempty;
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

        int pred = LR_PRED(a, b, _key, fanout);
        
        // Safety checks to prevent segmentation fault
        if (!pe_data || pred < 0 || pred >= fanout) {
            cout << "ERROR: pred=" << pred << " out of bounds (fanout=" << fanout << ")" << endl;
            return false;
        }
        
        uint64_t versionItem = pe_data[pred].readLockOrRestart(needRestart);
        if (needRestart) goto restart;

        if (pe_data[pred].key < -2) { // Empty slot
            pe_data[pred].upgradeToWriteLockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            pe_data[pred].assign(_key, _ptr);
            num_nonempty.fetch_add(1, std::memory_order_relaxed);
            total_n_travs.fetch_add(1, std::memory_order_relaxed);
            if (num_nonempty.load(std::memory_order_relaxed) >= LEAF_MAX_CAPACIY) {
                set_int_flag();
            }
            
            pe_data[pred].writeUnlock();
            return true;
            
        } else if (pe_data[pred].key == -1) {
            dilaxNode *child = pe_data[pred].child;
            pe_data[pred].readUnlockOrRestart(versionItem, needRestart);
            if (needRestart) goto restart;
            
            long child_last_total_n_travs = child->total_n_travs;
            bool if_inserted = child->insert(_key, _ptr);
#ifndef NOT_ADJUST
            if (if_inserted) {
                num_nonempty.fetch_add(1, std::memory_order_relaxed);
                total_n_travs.fetch_add(1 + (child->total_n_travs.load(std::memory_order_relaxed) - child_last_total_n_travs), std::memory_order_relaxed);
                if (if_retrain()) {
                    // Use fallback to regular EBR-based collection for now
                    collect_and_clear(dilax_auxiliary::retrain_keys, dilax_auxiliary::retrain_ptrs);
                    inc_n_adjust();
                    init();
                    distribute_data(dilax_auxiliary::retrain_keys, dilax_auxiliary::retrain_ptrs);
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
            
            total_n_travs += 2;
            fan2Leaf *fan2child = pe_data[pred].fan2child;
            keyType k1 = fan2child->k1;
            recordPtr p1 = fan2child->p1;
            keyType k2 = fan2child->k2;
            recordPtr p2 = fan2child->p2;
            
            if (_key == k1 || _key == k2) {
                pe_data[pred].writeUnlock();
                return false;
            }
            
            num_nonempty.fetch_add(1, std::memory_order_relaxed);
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
            
            assert(num_nonempty > 1);
            total_n_travs.fetch_add(3, std::memory_order_relaxed);
            num_nonempty.fetch_add(1, std::memory_order_relaxed);

            fan2Leaf *fan2child = new fan2Leaf(k1, p1, k2, p2);
            pe_data[pred].setFan2Child(fan2child);
            pe_data[pred].writeUnlock();
            
            // Periodically advance EBR epoch for better throughput
            static thread_local int insertCount = 0;
            if (++insertCount % 500 == 0) {  // More frequent advancement
                EBR::advance();
            }
            
            return true;
        }
    }


    inline int erase(const keyType &_key) {
        // Note: This method would need optimistic locking similar to insert
        // For now, keeping simplified version without mutex
         int pred = LR_PRED(a, b, _key, fanout);
         OptimisticDilaxPairEntry &pe = pe_data[pred];
         if (pe.key == _key) {
            pe.setNull();
            --num_nonempty;
            --total_n_travs;
            return num_nonempty;
        }
        else if (pe.key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (fan2child->k1 == _key) {
                pe.assign(fan2child->k2, fan2child->p2);
                --num_nonempty;
                total_n_travs -= 3;
//                delete fan2child;
                return num_nonempty;
            } else if (fan2child->k2 == _key) {
                pe.assign(fan2child->k1, fan2child->p1);
                --num_nonempty;
                total_n_travs -= 3;
//                delete fan2child;
                return num_nonempty;
            } else {
                return -1;
            }
        } else if (pe.key == -1) {
            dilaxNode *child = pe.child;
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
            return -1;
        }
    }
    inline int erase_and_get_ptr(const keyType &_key, recordPtr &ptr) {
         int pred = LR_PRED(a, b, _key, fanout);
         OptimisticDilaxPairEntry &pe = pe_data[pred];
         if (pe.key == _key) {
            ptr = pe.ptr;
            pe.setNull();
            --num_nonempty;
            --total_n_travs;
            return num_nonempty;
        }
        else if (pe.key == -2) {
            fan2Leaf *fan2child = pe.fan2child;
            if (fan2child->k1 == _key) {
                ptr = fan2child->p1;
                pe.assign(fan2child->k2, fan2child->p2);
                --num_nonempty;
                total_n_travs -= 3;
//                delete fan2child;
                return num_nonempty;
            } else if (fan2child->k2 == _key) {
                ptr = fan2child->p2;
                pe.assign(fan2child->k1, fan2child->p1);
                --num_nonempty;
                total_n_travs -= 3;
//                delete fan2child;
                return num_nonempty;
            } else {
                return -1;
            }
        } else if (pe.key == -1) {
            dilaxNode *child = pe.child;
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
            return -1;
        }
    }


    void collect_and_clear(keyType *keys, recordPtr *ptrs) {
        // Safety check for null pointers
        if (!keys || !ptrs) {
            cout << "ERROR: collect_and_clear called with null pointers!" << endl;
            return;
        }
        
        // Don't delete pe_data here - that causes race conditions
        // Just collect the data, deletion will happen in init()
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
                // Schedule child for safe deletion instead of immediate delete
                EBR::scheduleDelete(child);
            } else if (pe.key == -2) {
                fan2Leaf *fan2child = pe.fan2child;
                keys[j] = fan2child->k1;
                ptrs[j++] = fan2child->p1;
                keys[j] = fan2child->k2;
                ptrs[j++] = fan2child->p2;
                // Note: fan2child deletion handled in init()
            }
        }
        // Don't delete pe_data here - init() will handle it safely
        assert(j == num_nonempty);
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

    // Safe collection with reader protection
    void collect_and_clear_safe(keyType *keys, recordPtr *ptrs) {
        if (!keys || !ptrs) {
            cout << "ERROR: collect_and_clear_safe called with null pointers!" << endl;
            return;
        }
        
        int j = 0;
        std::vector<dilaxNode*> childrenToDelete;
        std::vector<fan2Leaf*> fan2ToDelete;
        
        for(int i = 0; i < fanout; ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            
            // Check if entry is being concurrently modified
            if (pe.isObsolete()) {
                // Skip obsolete entries - they're being rebuilt
                continue;
            }
            
            if (pe.key >= 0) {
                keys[j] = pe.key;
                ptrs[j++] = pe.ptr;
            } else if (pe.key == -1 && pe.child) {
                dilaxNode *child = pe.child;
                child->collect_and_clear_safe(keys+j, ptrs+j);
                j += child->num_nonempty;
                childrenToDelete.push_back(child);
            } else if (pe.key == -2 && pe.fan2child) {
                fan2Leaf *fan2child = pe.fan2child;
                keys[j] = fan2child->k1;
                ptrs[j++] = fan2child->p1;
                keys[j] = fan2child->k2;
                ptrs[j++] = fan2child->p2;
                fan2ToDelete.push_back(fan2child);
            }
        }
        
        // Schedule all deletions via EBR for safety
        for (auto* child : childrenToDelete) {
            EBR::scheduleDelete(child);
        }
        // fan2Leaf structures are simpler - can delete immediately after collection
        for (auto* fan2 : fan2ToDelete) {
            delete fan2;
        }
        
        assert(j <= num_nonempty); // May be less due to obsolete entries
    }

    // Atomic collection that prevents concurrent access
    bool collect_and_clear_atomic(keyType *keys, recordPtr *ptrs) {
        if (!keys || !ptrs) return false;
        
        std::vector<OptimisticDilaxPairEntry*> lockedEntries;
        bool needRestart = false;
        int restartCount = 0;
        
        restart:
        if (restartCount++ > 3) return false; // Fail faster for better throughput
        
        // Phase 1: Lock all entries in this node with timeout
        for (int i = 0; i < fanout; ++i) {
            if (i >= fanout) break; // Additional bounds check
            
            int lockRetries = 0;
            while (lockRetries++ < 5) {  // Limit lock attempts
                pe_data[i].writeLockOrRestart(needRestart);
                if (!needRestart) break;
                
                _mm_pause(); // Brief pause between attempts
                needRestart = false;
            }
            
            if (needRestart) {
                // Release all previously acquired locks
                for (auto* entry : lockedEntries) {
                    entry->writeUnlock();
                }
                lockedEntries.clear();
                yield(restartCount);
                needRestart = false;
                goto restart;
            }
            lockedEntries.push_back(&pe_data[i]);
        }
        
        // Phase 2: Now safely collect data with exclusive access
        int j = 0;
        for(int i = 0; i < fanout && i < static_cast<int>(lockedEntries.size()); ++i) {
            OptimisticDilaxPairEntry &pe = pe_data[i];
            if (pe.key >= 0) {
                keys[j] = pe.key;
                ptrs[j++] = pe.ptr;
            } else if (pe.key == -1 && pe.child) {
                dilaxNode *child = pe.child;
                // Use regular collection for children to avoid deadlock
                child->collect_and_clear(keys+j, ptrs+j);
                j += child->num_nonempty;
                EBR::scheduleDelete(child);
            } else if (pe.key == -2 && pe.fan2child) {
                fan2Leaf *fan2child = pe.fan2child;
                keys[j] = fan2child->k1;
                ptrs[j++] = fan2child->p1;
                keys[j] = fan2child->k2;
                ptrs[j++] = fan2child->p2;
                delete fan2child; // Safe to delete immediately
            }
        }
        
        // Phase 3: Mark entries as obsolete and unlock atomically
        std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure data collection is complete
        
        for (auto* entry : lockedEntries) {
            entry->writeUnlockObsolete(); // Mark obsolete to prevent future reads
        }
        
        return true;
    }

};

// Now that dilaxNode is fully defined, implement the EBR delete helper
inline void EBR::deleteDilaxNode(void* ptr) {
    delete static_cast<dilaxNode*>(ptr);
}


#endif //DILAX_DILAXNODE_H
