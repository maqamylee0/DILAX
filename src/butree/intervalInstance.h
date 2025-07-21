#include "interval.h"
#include "buInterval.h"
#include <string>
using namespace std;

#ifndef DILAX_INTERVALINSTANCE_H
#define DILAX_INTERVALINSTANCE_H

namespace dilax {

struct intervalInstance {
    static interval* newInstance(int type) {
        interval *i_ptr = new buInterval();
        return i_ptr;
    }

    static interval* newInstance(const string &type) {
        interval *i_ptr = new buInterval();
        return i_ptr;
    }
};

} // namespace dilax

#endif // DILAX_INTERVALINSTANCE_H
