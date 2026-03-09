#include "../include/map.h"

unsigned long long lcg_seed;

int distance(Pos a, Pos b) {
    int dy = abs(a.y - b.y);
    int dx;
    if (abs(a.y - b.y) % 2) {
        if (a.x > b.x)
            dx = std::max(0, abs(a.x - b.x) - abs(a.y - b.y) / 2 - (a.y % 2));
        else
            dx = std::max(0, abs(a.x - b.x) - abs(a.y - b.y) / 2 -
                                 (1 - (a.y % 2)));
    } else {
        dx = std::max(0, abs(a.x - b.x) - abs(a.y - b.y) / 2);
    }
    return dx + dy;
}

bool Map::is_empty(int x, int y, int player) const {
    if (x >= MAP_SIZE || y >= MAP_SIZE || x < 0 || y < 0)
        return false;
    if (map[x][y].base_camp != nullptr)
        return false;
    if (map[x][y].tower != nullptr)
        return false;
    return map[x][y].player == player;
}

bool Map::is_valid(int x, int y) const {
    if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE)
        return false;
    return map[x][y].valid;
}

const int map_L = SIDE_LENGTH;
Map::Map() {
    // 设置形状
    int k = 2 * map_L - 1;
    for (int i = map_L - 1; i >= 0; i--) {
        for (int j = 0; j < k; j++)
            map[(map_L - 1 - i) / 2 + j][i].valid = 1;
        k--;
    }
    k = 2 * map_L - 1;
    for (int i = map_L - 1; i <= 2 * map_L - 2; i++) {
        for (int j = 0; j < k; j++)
            map[(i - map_L + 1) / 2 + j][i].valid = 1;
        k--;
    }
    std::vector<std::vector<int>> valid_blocks = 
    {
        {6, 1}, {7, 1}, {9, 1}, {11, 1},
        {12, 1}, {4, 2}, {6, 2}, {8, 2},
        {9, 2}, {11, 2}, {13, 2}, {4, 3},
        {5, 3}, {13, 3}, {14, 3}, {6, 4}, 
        {8, 4}, {9, 4}, {11, 4}, {3, 5}, 
        {4, 5}, {7, 5}, {9, 5}, {11, 5}, 
        {14, 5}, {15, 5}, {3, 6}, {5, 6}, 
        {12, 6}, {14, 6}, {2, 7}, {5, 7}, 
        {6, 7}, {8, 7}, {9, 7}, {10, 7}, 
        {12, 7}, {13, 7}, {16, 7}, {1, 8}, 
        {2, 8}, {7, 8}, {10, 8}, {15, 8}, 
        {16, 8}, {0, 9}, {4, 9}, {5, 9}, 
        {6, 9}, {9, 9}, {12, 9}, {13, 9}, 
        {14, 9}, {18, 9}, {1, 10}, {2, 10}, 
        {7, 10}, {10, 10}, {15, 10}, {16, 10}, 
        {2, 11}, {5, 11}, {6, 11}, {8, 11}, 
        {9, 11}, {10, 11}, {12, 11}, {13, 11}, 
        {16, 11}, {3, 12}, {5, 12}, {12, 12}, 
        {14, 12}, {3, 13}, {4, 13}, {7, 13}, 
        {9, 13}, {11, 13}, {14, 13}, {15, 13}, 
        {6, 14}, {8, 14}, {9, 14}, {11, 14}, 
        {4, 15}, {5, 15}, {13, 15}, {14, 15}, 
        {4, 16}, {6, 16}, {8, 16}, {9, 16}, 
        {11, 16}, {13, 16}, {6, 17}, {7, 17}, 
        {9, 17}, {11, 17}, {12, 17}
    };
    for (auto block : valid_blocks) {
        map[block[0]][block[1]].valid = false;
    }
    // 划分player领地
    std::vector<std::vector<int>> tower_blocks = 
    {
        {6, 1}, {7, 1}, {4, 2}, {6, 2}, 
        {8, 2}, {4, 3}, {5, 3}, {6, 4}, 
        {8, 4}, {7, 5}, {5, 6}, {5, 7}, 
        {6, 7}, {8, 7}, {7, 8}, {4, 9}, 
        {5, 9}, {6, 9}, {7, 10}, {5, 11}, 
        {6, 11}, {8, 11}, {5, 12}, {7, 13}, 
        {6, 14}, {8, 14}, {4, 15}, {5, 15}, 
        {4, 16}, {6, 16}, {8, 16}, {6, 17}, 
        {7, 17}
    };
    for (auto block : tower_blocks) {
        map[block[0]][block[1]].player = 0;
    }
    tower_blocks = 
    {
        {11, 1}, {12, 1}, {9, 2}, {11, 2},
        {13, 2}, {13, 3}, {14, 3}, {9, 4}, 
        {11, 4}, {11, 5}, {12, 6}, {10, 7}, 
        {12, 7}, {13, 7}, {10, 8}, {12, 9}, 
        {13, 9}, {14, 9}, {10, 10}, {10, 11}, 
        {12, 11}, {13, 11}, {12, 12}, {11, 13}, 
        {9, 14}, {11, 14}, {13, 15}, {14, 15}, 
        {9, 16}, {11, 16}, {13, 16}, {11, 17}, 
        {12, 17}
    };
    for (auto block : tower_blocks) {
        map[block[0]][block[1]].player = 1;
    }

    // 初始化信息素
    // for(int i = 0; i < 2; i++)
    //     for(int j = 0; j < MAP_SIZE; j++)
    //         for(int k = 0; k < MAP_SIZE; k++)
    //             map[j][k].pheromone[i] = lcg() * pow(2, -46) + 8;
    // for (int i = 0; i < MAP_SIZE; i++)
    //     for (int j = 0; j < MAP_SIZE; j++) {
    //             map[i][j].pheromone[0] = TAU_BASE;
    //             map[i][j].pheromone[1] = TAU_BASE;
    //         }
}

void Map::build(DefenseTower *new_tower) {
    map[new_tower->get_x()][new_tower->get_y()].tower = new_tower;
}

bool Map::destroy(int x, int y) {
    map[x][y].tower = nullptr;
    return true;
}

unsigned long long lcg(){
    lcg_seed = (25214903917 * lcg_seed) & ((1ll << 48) - 1);
    return lcg_seed;
}
void Map::init_pheromon(unsigned long long M){
    lcg_seed = M;
    for(int i = 0; i < 2; i++)
        for(int j = 0; j < MAP_SIZE; j++)
            for(int k = 0; k < MAP_SIZE; k++) {
                unsigned long long v = lcg();
                map[j][k].pheromone[i] = 80000 + (int)((v * 10000ULL) >> 46);
            }
}
