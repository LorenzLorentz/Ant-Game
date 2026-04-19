#pragma once

enum ItemType {
    LightingStorm,
    EMPBlaster,
    Deflectors,
    EmergencyEvasion,
    Count,
};

int get_item_cd(ItemType type);

int get_item_time(ItemType type);

struct Item {
    // 武器cd
    int cd;
    // 武器持续时间
    int duration;
    // Internal-only guard to avoid re-triggering a just-deployed effect twice in one round.
    int last_trigger_round;
    int x, y;
    Item(int _cd, int _duration, int _x, int _y)
        : cd(_cd), duration(_duration), last_trigger_round(-1), x(_x), y(_y) {
    }
    Item(ItemType it, int _x, int _y)
        : cd(get_item_cd(it)),
          duration(get_item_time(it)),
          last_trigger_round(-1),
          x(_x),
          y(_y) {
    }
};
