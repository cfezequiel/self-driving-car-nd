#ifndef MAP_H
#define MAP_H

#include <vector>

using std::vector;


struct MapWaypoints {
  vector<double> x;
  vector<double> y;
  vector<double> s;
  vector<double> dx;
  vector<double> dy;
};

#endif  // MAP_H
