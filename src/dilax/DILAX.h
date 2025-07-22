#include "dilaxNode.h"
#include "../butree/interval_utils.h"
#include "../global/global.h"
#include "../utils/data_utils.h"
#include "../utils/linux_sys_utils.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>

// Performance optimization macros
#ifndef LIKELY
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
#include <stack>
#include <shared_mutex>
#include <mutex>
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstring>

// Branch prediction hints for better performance
#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif
#ifndef UNLIKELY  
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#ifndef DILAX_DILAX_H
#define DILAX_DILAX_H


namespace dilaxFunc {
    pair<dilaxNode **, double *>
    create_children(const int &height, dilaxNode **parents, int n_parents, double *parents_range_froms,
                    keyType *split_keys_for_children, recordPtr *ptrs, int n_keys,
                    int &act_total_N_children);
}

class DILAX {
    dilaxNode *root;
    string mirror_dir;
    
    // Minimal synchronization: only tree-level mutex and build flag
    mutable std::shared_mutex tree_mutex;       // For tree structure protection
    std::atomic<bool> is_built{false};          // Atomic flag to check if tree is built

public:
    //----for SOSD benchmark
    uint64_t Build(const std::vector< pair<keyType, recordPtr> >& data) {
        return linux_sys_utils::timing(
                [&] { bulk_load(data); });
    }

    DilaxSearchBound EqualityLookup(const keyType &lookup_key) const {
        const uint64_t start = search(lookup_key);
        const uint64_t stop = start + 1;

        return (DilaxSearchBound){start, stop};
    }

    std::string name() const { return "DILAX"; }

    std::size_t size() const { return total_size(); }

    // ---------------------
    DILAX(): root(NULL) {
        init_insert_aux_vars();
    }
    ~DILAX() {
        clear();
    }

    void clear() {
        // Use exclusive lock for clearing operation
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        
        is_built.store(false);
        
        if (root) {
            delete root;
            root = NULL;
        }
        free_insert_aux_vars();
    }


    void init_insert_aux_vars() {
        dilax_auxiliary::init_insert_aux_vars();
    }

    void free_insert_aux_vars() {
        dilax_auxiliary::free_insert_aux_vars();
    }

    void set_mirror_dir(const std::string &dir) { mirror_dir = dir; }

    void build_from_mirror(l_matrix &mirror, const keyArray &all_keys, const recordPtrArray &all_ptrs, long N) {
        // Use exclusive lock during tree construction
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        
        // Mark as not built during construction
        is_built.store(false);
        
        size_t H = mirror.size();

//        cout << "+++H = " << H << endl;
        intVec n_nodes_each_level;
        intVec n_nodes_each_level_mirror;
        for (longVec &lv : mirror) {
            n_nodes_each_level_mirror.push_back(lv.size());
        }

        keyType **split_keys_list = new keyType *[H];
        split_keys_list[0] = all_keys.get();

        for (int height = H - 1; height > 0; --height) {
            longVec &lv = mirror[height-1];
            int n_split_keys = lv.size();

            long *split_keys = new long[n_split_keys + 1];
            std::copy(lv.begin(), lv.end(), split_keys);
            split_keys[n_split_keys] = all_keys[N-1] + 1;

            split_keys_list[height] = split_keys;
            data_utils::check(split_keys, n_split_keys+1);
        }
        root = new dilaxNode(true);
//    root->set_range(0, all_keys[N-1] + 1);
        int n_keys = n_nodes_each_level_mirror[H - 2];
        root->fanout = n_keys + 1;
        long ubd = split_keys_list[H-1][n_keys-1];
        long lbd = split_keys_list[H-1][0];
        root->b = 1.0 * n_keys / (ubd - lbd);
        root->a = -(root->b * lbd);
        n_nodes_each_level.push_back(1);
//    root->children_init();
        root->pe_data = new dilaxPairEntry[root->fanout];

        keyType lastone = split_keys_list[H - 2][n_nodes_each_level_mirror[H-3]-1];

        dilaxNode **parents = new dilaxNode*[1];
        parents[0] = root;
        int n_parents = 1;
        double *parents_range_froms = new double[2];
        parents_range_froms[0] = 0;
        parents_range_froms[1] = all_keys[N-1] + 1;

        // height: the height of parents
        dilaxNode **children = NULL;
        int act_total_N_children = 0;

        for (int height = H - 1; height > 0; --height) {
            n_keys = N;

            if (height > 1) {
                n_keys = n_nodes_each_level_mirror[height-2];
            }
            recordPtr *ptrs = NULL;
            if (height == 1) {
                ptrs = all_ptrs.get();
            }

            pair<dilaxNode **, double *> _pair = dilaxFunc::create_children(height, parents, n_parents, parents_range_froms, split_keys_list[height-1],
                                                                          ptrs, n_keys, act_total_N_children);
            children = _pair.first;
            double *children_range_froms = _pair.second;
            delete[] parents;

            parents = children;

            delete[] parents_range_froms;
            parents_range_froms = children_range_froms;

            n_parents = act_total_N_children;
            n_nodes_each_level.push_back(act_total_N_children);
        }

        for (int height = 1; height < H; ++height) {
            delete[] split_keys_list[height];
        }
        delete[] split_keys_list;

        // Simplified sequential assignment of keys to leaves for better cache performance
        for (long i = 0; i < N; ++i) {
            dilaxNode *leaf = find_leaf(all_keys[i]);
            leaf->inc_num_nonempty();
        }

        // Sequential bulk loading of leaf nodes for better performance  
        long start_idx = 0;
        bool print = false;
        
        for (int i = 0; i < act_total_N_children; ++i) {
            dilaxNode *leaf = children[i];
            int _num_nonempty = leaf->num_nonempty;
            leaf->bulk_loading(all_keys.get() + start_idx, all_ptrs.get() + start_idx, print);
            start_idx += _num_nonempty;
        }
        
        if (start_idx != N) {
            cout << "error, start_idx = " << start_idx << ", N = " << N << endl;
        }
        assert(start_idx == N);

//        validness_check(all_keys, all_ptrs, N);

        root->trim();
        root->cal_num_nonempty();
#ifndef ALLOW_FAN2_NODE
        root->simplify();
#endif
        root->cal_avg_n_travs();
        root->init_after_bulk_load();
        
        // Mark tree as built after successful construction
        is_built.store(true);
    }

    size_t total_size() const{
        // Simple check without locking for read-only operation
        if (!is_built.load() || !root) {
            return 0;
        }
        
        std::stack<dilaxNode*> s;
        s.push(root);

        size_t size = 0;
        size_t delta = sizeof(dilaxNode);
        while (!s.empty()) {
            dilaxNode* node = s.top(); s.pop();

            size += delta;
            if (!(node->is_internal())) {
                size += node->num_nonempty * 2 * sizeof(long);
            } else {
                for (int i = 0; i < node->fanout; ++i) {
                    dilaxPairEntry &kp = node->pe_data[i];
                    if (kp.key < 0) {
                        if (kp.key == -1) {
                            s.push(kp.child);
                            size += sizeof(long);
                        } else if (kp.key == -2) {
                            size += (delta + 5 * sizeof(long));
                        }
                    } else {
                        size += 2 * sizeof(long);
                    }
                }
            }
        }
        return size;
    }


    void validness_check(keyType *keys, recordPtr *ptrs, int n_keys) {
        for (int i = 0; i < n_keys; ++i) {
            recordPtr pred = search(keys[i]);
            if (pred != ptrs[i]) {
                cout << "i = " << i << ", key = " << keys[i] << ", pred = " << pred << ", ptr = " << ptrs[i] << endl;
            }
            assert(pred == ptrs[i]);
        }
    }

    inline bool insert(const keyType &key, const recordPtr &ptr) { 
        if (UNLIKELY(!is_built.load(std::memory_order_acquire))) {
            return false;  // Tree not built yet
        }
        
        // Simple approach: shared lock for finding, exclusive for writing
        // This avoids complex node-level locking overhead
        std::unique_lock<std::shared_mutex> write_lock(tree_mutex);
        
        dilaxNode* target_node = find_leaf(key);
        if (target_node) {
            return target_node->insert(key, ptr);
        }
        
        return false;
    };
    
    inline bool insert(const pair<keyType, recordPtr> &p) { 
        return insert(p.first, p.second); 
    };
    inline bool erase(const keyType &key) { 
        if (UNLIKELY(!is_built.load(std::memory_order_acquire))) {
            return false;  // Tree not built yet
        }
        
        // Simple approach: exclusive lock for all writes
        std::unique_lock<std::shared_mutex> write_lock(tree_mutex);
        
        dilaxNode* target_node = find_leaf(key);
        if (target_node) {
            return target_node->erase(key) >= 0;
        }
        
        return false;
    };
    
    inline recordPtr delete_key(const keyType &key) {
        if (UNLIKELY(!is_built.load(std::memory_order_acquire))) {
            return -1;  // Tree not built yet
        }
        
        // Use exclusive lock for writes
        std::unique_lock<std::shared_mutex> write_lock(tree_mutex);
        
        // Find target node under exclusive lock
        dilaxNode* target_node = find_leaf(key);
        recordPtr ptr = static_cast<recordPtr>(-1);
        if (target_node) {
            target_node->erase_and_get_ptr(key, ptr);
        }
        return ptr;
    }

    void save(const string &path);
    void load(const string &path);
    dilaxNode* loadNode(FILE *fp);


    // Optimistic leaf finding that works under shared locks
    inline dilaxNode* find_leaf(const keyType &key) {
        dilaxNode *node = root;
        if (UNLIKELY(!node)) return nullptr;
        
        // Traverse down to leaf level with consistent predictions
        while (node && node->is_internal()) {
            if (UNLIKELY(!node->pe_data || node->fanout <= 0)) {
                return nullptr;
            }
            
            int pred = LR_PRED(node->a, node->b, key, node->fanout);
            if (UNLIKELY(pred < 0 || pred >= node->fanout)) {
                return nullptr;
            }
            
            dilaxPairEntry &kp = node->pe_data[pred];
            if (kp.key == -1) {
                node = kp.child;
            } else {
                // This is a leaf entry, break and return current node
                break;
            }
        }
        
        return node;
    }



    inline long search(const keyType &key) const {
        if (UNLIKELY(!is_built.load(std::memory_order_acquire))) {
            return -1;  // Tree not built yet
        }
        
        // RCU-style optimistic read: try lock-free first, fallback to shared lock
        dilaxNode *node = root;
        
        // Optimistic read attempt with memory barriers
        while (node) {
            // Read node version before accessing data
            uint32_t version_before = node->get_version();
            if (node->is_locked(version_before)) {
                break;  // Node is locked, fallback to shared lock
            }
            
            // Memory barrier before reading node data
            std::atomic_thread_fence(std::memory_order_acquire);
            
            if (UNLIKELY(!node->pe_data || node->fanout <= 0)) {
                break;  // Corrupted state, fallback to shared lock
            }
            
            int pred = LR_PRED(node->a, node->b, key, node->fanout);
            if (UNLIKELY(pred < 0 || pred >= node->fanout)) {
                break;  // Invalid prediction, fallback to shared lock
            }
            
            // Copy the pair entry atomically
            dilaxPairEntry kp = node->pe_data[pred];
            
            // Memory barrier after reading data
            std::atomic_thread_fence(std::memory_order_acquire);
            
            // Validate version hasn't changed
            uint32_t version_after = node->get_version();
            if (version_before != version_after || node->is_locked(version_after)) {
                break;  // Version changed, fallback to shared lock
            }
            
            // Process the entry
            if (kp.key == key) {
                return kp.ptr;
            } else if (kp.key == -1) {
                node = kp.child;
            } else if (kp.key == -2) {
                fan2Leaf *child = kp.fan2child;
                if (child && child->k1 == key) return child->p1;
                if (child && child->k2 == key) return child->p2;
                return -1;
            } else {
                return -1;
            }
        }
        
        // Fallback to shared lock for consistency
        std::shared_lock<std::shared_mutex> read_lock(tree_mutex);
        node = root;
        while (node) {
            if (UNLIKELY(!node->pe_data || node->fanout <= 0)) {
                return -1;
            }
            
            int pred = LR_PRED(node->a, node->b, key, node->fanout);
            if (UNLIKELY(pred < 0 || pred >= node->fanout)) {
                return -1;
            }
            
            dilaxPairEntry &kp = node->pe_data[pred];
            if (kp.key == key) {
                return kp.ptr;
            } else if (kp.key == -1) {
                node = kp.child;
            } else if (kp.key == -2) {
                fan2Leaf *child = kp.fan2child;
                if (child && child->k1 == key) return child->p1;
                if (child && child->k2 == key) return child->p2;
                return -1;
            } else {
                return -1;
            }
        }
        return -1;
    }


    inline int range_query(const keyType &k1, const keyType &k2, recordPtr *ptrs) { 
        if (UNLIKELY(!is_built.load(std::memory_order_acquire))) {
            return 0;  // Tree not built yet
        }
        
        // Use shared lock for range queries to ensure consistency while allowing concurrent reads
        std::shared_lock<std::shared_mutex> read_lock(tree_mutex);
        if (UNLIKELY(!root)) return 0;
        
        return root->range_query(k1, k2, ptrs);
    }


    void bulk_load(const keyArray &keys, const recordPtrArray &ptrs, long n_keys);//, const string &mirror_dir, const string &layout_conf_path, int interval_type=1);
    void bulk_load(const std::vector< pair<keyType, recordPtr> > &bulk_load_data);
};



#endif //DILAX_DILAX_H
