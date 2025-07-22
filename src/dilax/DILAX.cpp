#include "DILAX.h"
#include "../global/linearReg.h"
#include "../global/global.h"
#include "../utils/data_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <algorithm>
#include <functional>
#include <utility>
#include <cassert>
#include <thread>
#include <future>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <regex>
#include <cstdio>

#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

using namespace std;


//let i = returned value,  range_tos[i] <= key < range_tos[i+1]
namespace dilaxFunc {
    // Parallel version of create_children for larger workloads
    pair<dilaxNode **, double *>
    create_children_parallel(const int &height, dilaxNode **parents, int n_parents, double *parents_range_froms,
                            keyType *split_keys_for_children, recordPtr *ptrs, int n_keys,
                            int &act_total_N_children) {
        act_total_N_children = 0;
        for (int i = 0; i < n_parents; ++i) {
            act_total_N_children += parents[i]->get_fanout();
        }
        dilaxNode **_children = new dilaxNode *[act_total_N_children];

        double *children_range_froms = NULL;
        if (height > 1) {
            children_range_froms = new double[act_total_N_children + 1];
        }

        // Parallel processing for larger workloads
        const int parallel_threshold = 4; // Minimum number of parents to parallelize
        
        if (n_parents >= parallel_threshold && act_total_N_children >= 32) {
            // Process parents in parallel
            std::vector<std::future<void>> futures;
            std::atomic<int> cursor_atomic{0};
            std::vector<int> parent_cursors(n_parents);
            
            // Calculate cursor positions for each parent
            int temp_cursor = 0;
            for (int i = 0; i < n_parents; ++i) {
                parent_cursors[i] = temp_cursor;
                temp_cursor += parents[i]->get_fanout();
            }
            
            // Process each parent in parallel
            for (int parent_idx = 0; parent_idx < n_parents; ++parent_idx) {
                futures.emplace_back(std::async(std::launch::async, [=, &_children, &children_range_froms]() {
                    dilaxNode *parent = parents[parent_idx];
                    int fanout = parent->fanout;
                    parent->pe_data = new dilaxPairEntry[fanout];
                    double range_from = parents_range_froms[parent_idx];
                    double parent_range_to = parents_range_froms[parent_idx + 1];
                    
                    int local_cursor = parent_cursors[parent_idx];
                    int last_idx = 0;
                    
                    // Calculate last_idx start position for this parent
                    for (int prev_i = 0; prev_i < parent_idx; ++prev_i) {
                        last_idx += parents[prev_i]->get_fanout();
                    }
                    
                    for (int child_id = 0; child_id < fanout; ++child_id) {
                        double range_to = (1.0 * (child_id + 1) - parent->a) / parent->b;
                        if (range_to > parent_range_to) {
                            assert(child_id >= (fanout - 1));
                            range_to = parent_range_to;
                        }
                        if (child_id == fanout - 1) {
                            range_to = parent_range_to;
                        }
                        
                        int idx = data_utils::array_lower_bound(split_keys_for_children, range_to, 0, n_keys);
                        int n_keys_this_child = idx - last_idx;
                        
                        dilaxNode *child = NULL;
                        if (height > 1) {
                            dilaxNode *int_node = new dilaxNode(true);
                            if (n_keys_this_child > 0) {
                                int_node->cal_lr_params(split_keys_for_children + last_idx, n_keys_this_child);
                            }
                            child = int_node;
                            if (children_range_froms) {
                                children_range_froms[local_cursor] = range_from;
                            }
                        } else {
                            child = new dilaxNode(false);
                        }
                        
                        _children[local_cursor++] = child;
                        parent->pe_data[child_id].setChild(child);
                        
                        last_idx = idx;
                        range_from = range_to;
                    }
                }));
            }
            
            // Wait for all parallel tasks to complete
            for (auto& future : futures) {
                future.wait();
            }
            
            if (height > 1 && children_range_froms) {
                children_range_froms[act_total_N_children] = parents_range_froms[n_parents];
            }
        } else {
            // Fall back to sequential processing for smaller workloads
            return create_children(height, parents, n_parents, parents_range_froms,
                                 split_keys_for_children, ptrs, n_keys, act_total_N_children);
        }
        
        return make_pair(_children, children_range_froms);
    }

    pair<dilaxNode **, double *>
    create_children(const int &height, dilaxNode **parents, int n_parents, double *parents_range_froms,
                    keyType *split_keys_for_children, recordPtr *ptrs, int n_keys,
                    int &act_total_N_children) {
        act_total_N_children = 0;
        for (int i = 0; i < n_parents; ++i) {
            act_total_N_children += parents[i]->get_fanout();
        }
        dilaxNode **_children = new dilaxNode *[act_total_N_children];

        double *children_range_froms = NULL;
        if (height > 1) {
            children_range_froms = new double[act_total_N_children + 1];
        }

        dilaxNode *child = NULL;
        int last_idx = 0;
        int cursor = 0;

//        cout << "****height = " << height << ", n_parents = " << n_parents << ", n_keys = " << n_keys << endl;

        for (int i = 0; i < n_parents; ++i) {
            dilaxNode *parent = parents[i];
            int fanout = parent->fanout;
            parent->pe_data = new dilaxPairEntry[fanout];
            double range_from = parents_range_froms[i];
            double parent_range_to = parents_range_froms[i + 1];
//        parent->children = new dilaxNode*[fanout];
//        double range_from = parent->range_from;

            for (int child_id = 0; child_id < fanout; ++child_id) {
                double range_to = (1.0 * (child_id + 1) - parent->a) / parent->b;
                if (range_to > parent_range_to) {
                    assert(child_id >= (fanout - 1));
                    range_to = parent_range_to;
                }
                if (child_id == fanout - 1) {
                    range_to = parent_range_to;
                }
                int idx = data_utils::array_lower_bound(split_keys_for_children, range_to, 0, n_keys);
                int n_keys_this_child = idx - last_idx;

                if (last_idx > idx) {
                    cout << "child_id = " << child_id << ", range_from = " << range_from << ", range_to = " << range_to
                         << ", parent_range_to = " << parent_range_to << endl;
                }
                assert(idx >= last_idx);

                if (height > 1) {
                    dilaxNode *int_node = new dilaxNode(true);
//                int_node->set_range(range_from, range_to);
                    int_node->cal_lr_params(split_keys_for_children + last_idx, n_keys_this_child);
//                int_node->children_init();
                    child = int_node;
                    children_range_froms[cursor] = range_from;
                    _children[cursor++] = child;
                } else {
                    child = new dilaxNode(false);
                    _children[cursor++] = child;
                }
//            parent->children[child_id] = child;
                parent->pe_data[child_id].setChild(child);

                last_idx = idx;
                range_from = range_to;
            }

//        std::copy(parent->children, parent->children + fanout, _children + cursor);
//        cursor += fanout;
        }
        if (height > 1) {
            children_range_froms[cursor] = parents_range_froms[n_parents];
        }

        assert(last_idx = n_keys);
        return make_pair(_children, children_range_froms);
    }
}

void DILAX::save(const string &path) {
    FILE *fp = NULL;

    if (NULL == (fp = fopen(path.c_str(), "wb"))) {
        cout << path << " cannot be created." << endl;
        exit(1);
    }

//    saveNode(root, fp);
    root->save(fp);
    fclose(fp);
}

void DILAX::load(const string &path) {
    FILE *fp = NULL;

    if (NULL == (fp = fopen(path.c_str(), "rb"))) {
        cout << path << " cannot be opened." << endl;
        exit(1);
    }
    root = new dilaxNode(true);
    root->load(fp);
    fclose(fp);
}

void DILAX::bulk_load(const keyArray &keys, const recordPtrArray &ptrs, long n_keys) { //}, const string &mirror_dir, const string &layout_conf_path, int interval_type) {
    const int interval_type = 1;
    l_matrix mirror;
    dilax::build_ideal_mirror(keys, nullptr, n_keys, mirror, mirror_dir, interval_type);
//    build_mirror(keys, nullptr, n_keys, mirror, mirror_dir, interval_type);

//    cout << "----mirror.layout:------" << endl;
//    for (size_t i = 0; i < mirror.size(); ++i) {
//        cout << mirror[i].size() << " " << endl;
//    }
//    cout << endl;

    cout << "Building " << name() << "......" << endl;
    build_from_mirror(mirror, keys, ptrs, n_keys);
}

void DILAX::bulk_load(const std::vector< pair<keyType, recordPtr> > &bulk_load_data) {
    size_t N = bulk_load_data.size();
    keyArray keys = std::make_unique<keyType []>(N + 1);
    recordPtrArray ptrs = std::make_unique<recordPtr []>(N + 1);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = bulk_load_data[i].first;
        ptrs[i] = bulk_load_data[i].second;
    }
    keys[N] = keys[N-1] + 1;
    ptrs[N] = -1;
    bulk_load(keys, ptrs, static_cast<long>(N));
}
