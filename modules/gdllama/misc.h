
#ifndef MISC_H
#define MISC_H

#include <string>
#include "core/object/ref_counted.h"

std::string string_gd_to_std(String s);
String string_std_to_gd(std::string s);

bool is_utf8(const char * string);

#endif
