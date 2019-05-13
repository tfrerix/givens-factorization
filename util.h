#include <string>

#ifndef HELPER_H
#define HELPER_H

// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);

#endif
