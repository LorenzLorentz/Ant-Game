#include "../include/item.h"

// get cd of item
int get_item_cd(ItemType type) {
    int cd[4] = {35, 45, 25, 25};
    return cd[type];
}

// get duration of item
int get_item_time(ItemType type) {
    int time[4] = {15, 10, 10, 1};
    return time[type];
}
