#pragma once

#include "H5Cpp.h"


void refine_between_solutions(H5::H5File &data, const int index1, const int index2, const double dsabs);
