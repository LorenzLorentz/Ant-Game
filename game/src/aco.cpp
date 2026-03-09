#include "../include/map.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <vector>

// Pheromone deltas (scaled by PHEROMONE_SCALE)
const int Q1_INT = 100000;   // +10
const int Q2_INT = -50000;   // -5
const int Q3_INT = -30000;   // -3

// y 为奇偶时方向不同
const int d[2][6][2] = {{{0, 1}, {-1, 0}, {0, -1}, {1, -1}, {1, 0}, {1, 1}},
                        {{-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, 0}, {0, 1}}};

// Eta weight scaled by 10000 for integer arithmetic
int eta_scaled(int _x, int _y, int x, int y, Pos des) {
    if (distance(Pos(_x, _y), des) < distance(Pos(x, y), des)) {
        return 12500;   // 1.25
    } else if (distance(Pos(_x, _y), des) == distance(Pos(x, y), des)) {
        return 10000;   // 1.0
    } else {
        return 7500;    // 0.75
    }
}

// void Map::update_move_pheromone(Ant *ant) {
//     int L_k = std::max(ant->get_path_len(), 1);
//     int mov = -1;
//     if ((!ant->path.empty()) && (ant->get_status() != Ant::Status::Frozen))
//         mov = *(ant->path.end() - 1);
//     int player = ant->get_player();
//     int x = ant->get_x();
//     int y = ant->get_y();
//     // 移动信息素变化
//     if (mov != -1) {
//         map[x][y].pheromone[player][mov] += (double)Q0 / L_k;
//         x = x + d[y % 2][mov][0];
//         y = y + d[y % 2][mov][1];
//         map[x][y].pheromone[player][(mov + 3) % 6] += (double)Q0 / L_k;
//     }
//     ant->move(mov);
// }

// update pheromone
void Map::update_pheromone(Ant *ant) {
    int player = ant->get_player();
    // 如果到达大本营, 更新全局信息素
    // 如果hp <= 0 或已经走了很长距离, 判定死亡,更新全局信息素并返回
    int Q = 0;
    if (ant->get_status() == Ant::Status::Success) {
        Q = Q1_INT;
    } else if (ant->get_status() == Ant::Status::Fail) {
        Q = Q2_INT;
    } else if (ant->get_status() == Ant::Status::TooOld) {
        Q = Q3_INT;
    } else {
        return;
    }

    std::vector<Pos> trail = ant->get_trail_cells();
    if (trail.empty() ||
        !(trail.back() == Pos(ant->get_x(), ant->get_y()))) {
        trail.emplace_back(ant->get_x(), ant->get_y());
    }

    std::vector<std::pair<int, int>> visited_p;
    for (auto iter = trail.rbegin(); iter != trail.rend(); ++iter) {
        int x = iter->x;
        int y = iter->y;
        if (!is_valid(x, y))
            continue;
        if (std::find(visited_p.begin(), visited_p.end(), std::make_pair(x, y)) != visited_p.end())
            continue;
        map[x][y].pheromone[player] = std::max(TAU_MIN_INT, map[x][y].pheromone[player] + Q);
        visited_p.push_back(std::make_pair(x, y));
    }
}

// get the next step of ant
int Map::get_move(Ant *ant, Pos des) {
    int x = ant->get_x();
    int y = ant->get_y();
    int player = ant->get_player();

    long long weighted[6];
    for (int i = 0; i < 6; i++) {
        int _x = x + d[y % 2][i][0];
        int _y = y + d[y % 2][i][1];
        if (ant->get_last_move() >= 0 &&
            ant->get_last_move() == ((i + 3) % 6)) {
            weighted[i] = -1;
        } else if (!is_valid(_x, _y)) {
            weighted[i] = -1;
        } else {
            int eta = eta_scaled(_x, _y, x, y, des);
            weighted[i] = (long long)map[_x][_y].pheromone[player] * eta / PHEROMONE_SCALE;
        }
    }
    int mov = -1;
    long long max_p = -1;
    for (int i = 0; i < 6; i++) {
        if (weighted[i] > max_p) {
            max_p = weighted[i];
            mov = i;
        }
    }
    return mov;
}


// global attenuation: p_new = 0.97*p + 0.03*10
void Map::next_round() {
    for (int i = 0; i < MAP_SIZE; i++)
        for (int j = 0; j < MAP_SIZE; j++)
            for (int k = 0; k < 2; k++) {
                int p = map[i][j].pheromone[k];
                map[i][j].pheromone[k] = std::max(TAU_MIN_INT,
                    (LAMBDA_NUM * p + 3000 + 50) / LAMBDA_DENOM);
            }
}
