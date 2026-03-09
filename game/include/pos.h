#ifndef __POS_H__
#define __POS_H__

struct Pos {
    int x, y;
    Pos(){};
    Pos(int _x, int _y) : x(_x), y(_y){};
    bool operator==(const Pos &other) const {
        return (other.x == x) && (other.y == y);
    };
};

#endif
