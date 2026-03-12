#include "../include/game.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
using json = nlohmann::json;

// type of tower behaviors
#define TOWER_DESTROY_TYPE -1
#define TOWER_BUILD_TYPE 0
#define TOWER_UPGRADE_TYPE 1
#define TOWER_ATTACK_TYPE 2
// max coordinates of map
#define INIT_CAMP_HP 50
// max level of defensive tower
#define TOWER_MAX_LEVEL 2
// type of barrack behaviors
#define BARRACK_DESTROY_TYPE -1
#define BARRACK_BUILD_TYPE 0
// #define MAX_TIME 10
namespace {
constexpr unsigned long long RNG_MASK = (1ULL << 48) - 1;
constexpr unsigned long long RNG_MULTIPLIER = 25214903917ULL;
constexpr unsigned long long RNG_INCREMENT = 11ULL;
constexpr int RANDOM_FLOAT_BITS = 24;
constexpr double DEFAULT_MOVE_TEMPERATURE = 1.75;
constexpr double BEWITCH_MOVE_TEMPERATURE = 1.5;
constexpr double CROWDING_PENALTY = 1.25;
constexpr int RANDOM_ANT_DECAY_TURNS = 5;
constexpr int ANT_TELEPORT_INTERVAL = 10;
constexpr double ANT_TELEPORT_RATIO = 0.2;
constexpr double STALL_MOVE_PENALTY = 0.35;
constexpr double RETREAT_MOVE_PENALTY = 0.8;
constexpr double TARGET_PULL_DISTANCE_SCALE = 0.18;
constexpr int TOWER_KAMIKAZE_HP_THRESHOLD = 5;
constexpr double SPAWN_BEHAVIOR_PROBS[4] = {0.4, 0.3, 0.15, 0.15};
struct SpawnProfile {
    Ant::Kind kind;
    Ant::Behavior behavior;
};
constexpr SpawnProfile SPAWN_PROFILES[4] = {
    {Ant::Kind::Worker, Ant::Behavior::Default},
    {Ant::Kind::Worker, Ant::Behavior::Conservative},
    {Ant::Kind::Worker, Ant::Behavior::Randomized},
    {Ant::Kind::Combat, Ant::Behavior::Default},
};
constexpr std::size_t MAX_JUDGER_PACKET_SIZE = 16 * 1024 * 1024;
const int ant_dx[2][6][2] = {
    {{0, 1}, {-1, 0}, {0, -1}, {1, -1}, {1, 0}, {1, 1}},
    {{-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, 0}, {0, 1}},
};

std::uint32_t read_be_u32(const std::string &input) {
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
        value = (value << 8) + static_cast<unsigned char>(input[index]);
    }
    return value;
}

bool try_parse_json_payload(const std::string &input, json &parsed) {
    auto try_parse = [&parsed](const std::string &candidate) {
        try {
            parsed = json::parse(candidate);
            return true;
        } catch (const std::exception &) {
            return false;
        }
    };

    if (try_parse(input)) {
        return true;
    }
    if (input.size() >= 4) {
        std::uint32_t declared = read_be_u32(input);
        if (declared == input.size() - 4 && try_parse(input.substr(4))) {
            return true;
        }
    }

    std::size_t start = input.find_first_of("{[");
    if (start != std::string::npos && start != 0 &&
        try_parse(input.substr(start))) {
        return true;
    }
    return false;
}
} // namespace

unsigned long long Game::next_random() {
    rng_state = (RNG_MULTIPLIER * rng_state + RNG_INCREMENT) & RNG_MASK;
    return rng_state;
}

double Game::random_float() {
    return static_cast<double>((next_random() >> (48 - RANDOM_FLOAT_BITS)) &
                               ((1ULL << RANDOM_FLOAT_BITS) - 1ULL)) /
           static_cast<double>(1ULL << RANDOM_FLOAT_BITS);
}

int Game::random_index(int bound) {
    if (bound <= 1)
        return 0;
    return static_cast<int>(next_random() % static_cast<unsigned long long>(bound));
}

bool Game::ant_can_walk_to(int x, int y) const {
    if (!map.is_valid(x, y))
        return false;
    if (x == PLAYER_0_BASE_CAMP_X && y == PLAYER_0_BASE_CAMP_Y)
        return true;
    if (x == PLAYER_1_BASE_CAMP_X && y == PLAYER_1_BASE_CAMP_Y)
        return true;
    return map.map[x][y].player == -1;
}

bool Game::ant_can_target_cell(const Ant &ant, int x, int y) const {
    if (ant_can_walk_to(x, y))
        return true;
    const DefenseTower *tower = enemy_tower_at(ant.get_player(), x, y);
    return tower != nullptr;
}

DefenseTower *Game::enemy_tower_at(int player, int x, int y) {
    if (!map.is_valid(x, y))
        return nullptr;
    DefenseTower *tower = map.map[x][y].tower;
    if (tower == nullptr || tower->destroy() || tower->get_player() == player)
        return nullptr;
    return tower;
}

const DefenseTower *Game::enemy_tower_at(int player, int x, int y) const {
    if (!map.is_valid(x, y))
        return nullptr;
    const DefenseTower *tower = map.map[x][y].tower;
    if (tower == nullptr || tower->destroy() || tower->get_player() == player)
        return nullptr;
    return tower;
}

double Game::crowding_penalty(const Ant &ant, int x, int y) const {
    double penalty = 0.0;
    for (const auto &other : ants) {
        if (other.get_id() == ant.get_id() || other.get_player() != ant.get_player() ||
            other.get_status() == Ant::Status::Fail || other.get_status() == Ant::Status::TooOld)
            continue;
        int dist = distance(Pos(x, y), Pos(other.get_x(), other.get_y()));
        if (dist == 0)
            penalty += 1.0;
        else if (dist == 1)
            penalty += 0.35;
    }
    return penalty;
}

double Game::move_progress_score(const Ant &ant, int x, int y,
                                 const Pos &target) const {
    int current_distance = distance(Pos(ant.get_x(), ant.get_y()), target);
    int next_distance = distance(Pos(x, y), target);
    double score = static_cast<double>(current_distance - next_distance);
    if (next_distance == current_distance)
        score -= STALL_MOVE_PENALTY;
    else if (next_distance > current_distance)
        score -= RETREAT_MOVE_PENALTY *
                 static_cast<double>(next_distance - current_distance);
    int base_distance = distance(Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y),
                                 Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y));
    score += std::max(0.0, static_cast<double>(base_distance - next_distance)) *
             TARGET_PULL_DISTANCE_SCALE;
    return score;
}

double Game::move_pheromone_score(const Ant &ant, int x, int y) const {
    return static_cast<double>(map.map[x][y].pheromone[ant.get_player()]) /
           PHEROMONE_SCALE;
}

double Game::expected_damage_cost(const Ant &ant, int x, int y) const {
    double total = 0.0;
    int effective_hp = std::max(ant.get_hp(), 1);
    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.get_player() == ant.get_player() ||
            tower.is_producer())
            continue;
        if (distance(Pos(x, y), Pos(tower.get_x(), tower.get_y())) <=
            tower.get_range()) {
            total += static_cast<double>(tower.get_damage()) / effective_hp;
        }
    }
    return total;
}

double Game::control_risk_cost(const Ant &ant, int x, int y) const {
    if (ant.is_control_immune())
        return 0.0;
    double total = 0.0;
    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.get_player() == ant.get_player() ||
            tower.is_producer())
            continue;
        if (distance(Pos(x, y), Pos(tower.get_x(), tower.get_y())) >
            tower.get_range())
            continue;
        switch (tower.get_type()) {
        case TowerType::Ice:
            total += 1.0;
            break;
        case TowerType::Cannon:
            total += 1.3;
            break;
        case TowerType::Pulse:
            total += 0.7;
            break;
        default:
            break;
        }
    }
    return total;
}

double Game::tower_pull_score(const Ant &ant, int x, int y,
                              const DefenseTower *tower_target) const {
    if (tower_target != nullptr) {
        double bonus = ant.is_combat_ant() ? 6.0 : 1.5;
        bonus += std::max(0, TOWER_KAMIKAZE_HP_THRESHOLD - tower_target->get_hp()) *
                 0.75;
        return bonus;
    }
    if (!ant.is_combat_ant())
        return 0.0;
    double best = 0.0;
    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.get_player() == ant.get_player())
            continue;
        double distance_score =
            std::max(0.0, 6.0 - distance(Pos(x, y), Pos(tower.get_x(), tower.get_y())));
        double hp_score =
            std::max(0, TOWER_KAMIKAZE_HP_THRESHOLD - tower.get_hp()) * 0.6;
        best = std::max(best, distance_score + hp_score);
    }
    return best;
}

bool Game::ant_in_own_half(const Ant &ant) const {
    Pos own = ant.get_player() ? Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y)
                               : Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y);
    Pos enemy = ant.get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                 : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
    return distance(Pos(ant.get_x(), ant.get_y()), own) <=
           distance(Pos(ant.get_x(), ant.get_y()), enemy);
}

std::pair<int, int> Game::random_own_half_target(int player) {
    std::vector<std::pair<int, int>> cells;
    Pos own = player ? Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y)
                     : Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y);
    Pos enemy = player ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                       : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
    for (int x = 0; x < MAP_SIZE; ++x)
        for (int y = 0; y < MAP_SIZE; ++y)
            if (ant_can_walk_to(x, y) && !(x == own.x && y == own.y) &&
                distance(Pos(x, y), own) <= distance(Pos(x, y), enemy))
                cells.emplace_back(x, y);
    if (cells.empty())
        return {own.x, own.y};
    return cells[random_index(static_cast<int>(cells.size()))];
}

void Game::apply_control(Ant &ant, Ant::Behavior behavior,
                         const std::pair<int, int> *target) {
    if (ant.is_control_immune())
        return;
    ant.set_behavior(behavior);
    if (behavior == Ant::Behavior::Bewitched && target != nullptr)
        ant.set_bewitch_target(target->first, target->second);
}

void Game::maybe_control_free(Ant &ant, bool was_active, bool is_active) {
    if (was_active && !is_active && ant.get_behavior() != Ant::Behavior::ControlFree)
        ant.set_behavior(Ant::Behavior::ControlFree);
}

void Game::grant_emergency_evasion(Ant &ant, int stacks,
                                   bool grant_control_free_on_deplete) {
    ant.grant_evasion(stacks, grant_control_free_on_deplete);
}

void Game::prepare_ants_for_attack() {
    for (auto &ant : ants) {
        if (ant.is_frozen) {
            ant.is_frozen = false;
            if (ant.has_pending_behavior) {
                apply_control(ant, ant.pending_behavior);
                ant.clear_pending_behavior();
            }
        }
        bool current_deflect = false;
        Item deflect = item[ant.get_player()][ItemType::Deflectors];
        if (deflect.duration &&
            distance(Pos(ant.get_x(), ant.get_y()), Pos(deflect.x, deflect.y)) <= 3)
            current_deflect = true;
        bool current_evasion = false;
        Item evasion = item[ant.get_player()][ItemType::EmergencyEvasion];
        if (evasion.duration &&
            distance(Pos(ant.get_x(), ant.get_y()), Pos(evasion.x, evasion.y)) <= 3)
            current_evasion = true;
        maybe_control_free(ant, ant.defend, current_deflect);
        ant.defend = current_deflect;
        if (current_evasion)
            grant_emergency_evasion(ant, 2, true);
        ant.evasion = ant.shield > 0;
    }
}

void Game::damage_ant_by_tower(DefenseTower &tower, Ant &ant) {
    ant.set_hp(-tower.get_damage());
    if (ant.get_status() == Ant::Status::Fail)
        return;
    switch (tower.get_type()) {
    case TowerType::Ice:
        if (!ant.is_control_immune()) {
            ant.is_frozen = true;
            ant.set_pending_behavior_to(Ant::Behavior::Randomized);
        }
        break;
    case TowerType::Cannon:
        if (!ant.is_control_immune()) {
            std::pair<int, int> target = ant_in_own_half(ant)
                                             ? std::make_pair(ant.get_player() ? PLAYER_1_BASE_CAMP_X
                                                                               : PLAYER_0_BASE_CAMP_X,
                                                              ant.get_player() ? PLAYER_1_BASE_CAMP_Y
                                                                               : PLAYER_0_BASE_CAMP_Y)
                                             : random_own_half_target(ant.get_player());
            apply_control(ant, Ant::Behavior::Bewitched, &target);
        }
        break;
    case TowerType::Pulse:
        apply_control(ant, Ant::Behavior::Randomized);
        break;
    default:
        break;
    }
}

int Game::choose_ant_move(const Ant &ant) {
    Pos target = ant.get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                  : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
    bool allow_backtrack = ant.get_behavior() == Ant::Behavior::Randomized ||
                           ant.get_behavior() == Ant::Behavior::Bewitched;
    std::vector<std::tuple<int, int, int>> candidates;
    auto collect = [&](bool allow_reverse) {
        candidates.clear();
        for (int direction = 0; direction < 6; ++direction) {
            int nx = ant.get_x() + ant_dx[ant.get_y() % 2][direction][0];
            int ny = ant.get_y() + ant_dx[ant.get_y() % 2][direction][1];
            if (!allow_reverse && ant.get_last_move() >= 0 &&
                ant.get_last_move() == ((direction + 3) % 6))
                continue;
            if (!ant_can_target_cell(ant, nx, ny))
                continue;
            candidates.emplace_back(direction, nx, ny);
        }
    };
    collect(allow_backtrack);
    if (candidates.empty() && !allow_backtrack)
        collect(true);
    if (candidates.empty())
        return -1;
    if (ant.get_behavior() == Ant::Behavior::Randomized)
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);

    std::vector<double> scores;
    std::vector<double> raw_scores;
    scores.reserve(candidates.size());
    raw_scores.reserve(candidates.size());
    if (ant.get_behavior() == Ant::Behavior::Bewitched && ant.target_x >= 0 &&
        ant.target_y >= 0) {
        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            int eval_x = tower_target ? ant.get_x() : nx;
            int eval_y = tower_target ? ant.get_y() : ny;
            double score =
                ant.move_weights.progress *
                    move_progress_score(ant, eval_x, eval_y,
                                        Pos(ant.target_x, ant.target_y)) +
                ant.move_weights.pheromone *
                    move_pheromone_score(ant, eval_x, eval_y) -
                ant.move_weights.crowding *
                    crowding_penalty(ant, eval_x, eval_y) -
                ant.move_weights.expected_damage *
                    expected_damage_cost(ant, eval_x, eval_y) -
                ant.move_weights.control_risk *
                    control_risk_cost(ant, eval_x, eval_y) +
                ant.move_weights.tower_pull *
                    tower_pull_score(ant, eval_x, eval_y, tower_target) +
                (tower_target ? 4.0 : 0.0);
            scores.push_back(score);
            raw_scores.push_back(score);
        }
    } else {
        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            int eval_x = tower_target ? ant.get_x() : nx;
            int eval_y = tower_target ? ant.get_y() : ny;
            double progress = move_progress_score(ant, eval_x, eval_y, target);
            double pheromone = move_pheromone_score(ant, eval_x, eval_y);
            double tower_pull = tower_pull_score(ant, eval_x, eval_y, tower_target);
            double raw = progress + pheromone + tower_pull;
            raw_scores.push_back(raw);
            scores.push_back(
                ant.move_weights.progress * progress +
                ant.move_weights.pheromone * pheromone -
                ant.move_weights.crowding * crowding_penalty(ant, eval_x, eval_y) -
                ant.move_weights.expected_damage * expected_damage_cost(ant, eval_x, eval_y) -
                ant.move_weights.control_risk * control_risk_cost(ant, eval_x, eval_y) +
                ant.move_weights.tower_pull * tower_pull);
        }
    }
    if (ant.get_behavior() == Ant::Behavior::Conservative ||
        ant.get_behavior() == Ant::Behavior::ControlFree) {
        int best = 0;
        for (int i = 1; i < static_cast<int>(scores.size()); ++i)
            if (scores[i] > scores[best] ||
                (scores[i] == scores[best] && raw_scores[i] > raw_scores[best]))
                best = i;
        return std::get<0>(candidates[best]);
    }
    double temperature = ant.get_behavior() == Ant::Behavior::Bewitched
                             ? BEWITCH_MOVE_TEMPERATURE
                             : DEFAULT_MOVE_TEMPERATURE;
    double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> probs(scores.size(), 0.0);
    double total = 0.0;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        probs[i] = std::exp((scores[i] - max_score) / temperature);
        total += probs[i];
    }
    if (total <= 0.0)
        return std::get<0>(candidates[0]);
    double threshold = random_float();
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cumulative += probs[i] / total;
        if (threshold <= cumulative)
            return std::get<0>(candidates[i]);
    }
    return std::get<0>(candidates.back());
}

void Game::attack_tower_from_ant(Ant &ant, DefenseTower &tower) {
    if (ant.is_combat_ant() && tower.get_hp() < TOWER_KAMIKAZE_HP_THRESHOLD) {
        tower.take_damage(tower.get_hp());
        tower.set_changed_this_round();
        map.destroy(tower.get_x(), tower.get_y());
        tower.set_destroy();
        ant.set_hp_true(-ant.get_hp());
        return;
    }
    if (tower.take_damage(ant.get_tower_attack_damage())) {
        tower.set_changed_this_round();
        map.destroy(tower.get_x(), tower.get_y());
        tower.set_destroy();
    } else {
        tower.set_changed_this_round();
    }
}

void Game::resolve_ant_step(Ant &ant, int move) {
    if (move == Ant::NoMove) {
        ant.move(move);
        return;
    }
    int nx = ant.get_x() + ant_dx[ant.get_y() % 2][move][0];
    int ny = ant.get_y() + ant_dx[ant.get_y() % 2][move][1];
    DefenseTower *tower = enemy_tower_at(ant.get_player(), nx, ny);
    if (tower != nullptr) {
        attack_tower_from_ant(ant, *tower);
        ant.reset_backtrack();
        ant.evasion = ant.shield > 0;
        return;
    }
    ant.move(move);
}

void Game::teleport_ants() {
    if (ANT_TELEPORT_INTERVAL <= 0 || (round + 1) % ANT_TELEPORT_INTERVAL != 0)
        return;
    std::vector<Ant *> eligible;
    for (auto &ant : ants)
        if (ant.get_status() != Ant::Status::Fail &&
            ant.get_status() != Ant::Status::TooOld &&
            ant.get_behavior() != Ant::Behavior::ControlFree)
            eligible.push_back(&ant);
    if (eligible.empty())
        return;
    int teleport_count =
        std::max(1, static_cast<int>(std::round(eligible.size() * ANT_TELEPORT_RATIO)));
    while (!eligible.empty() && teleport_count-- > 0) {
        int ant_idx = random_index(static_cast<int>(eligible.size()));
        Ant *ant = eligible[ant_idx];
        eligible.erase(eligible.begin() + ant_idx);
        std::vector<std::pair<int, int>> cells;
        Pos own = ant->get_player() ? Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y)
                                    : Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y);
        Pos enemy = ant->get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                      : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
        if (ant_in_own_half(*ant)) {
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    if (ant_can_walk_to(x, y) &&
                        !(x == PLAYER_0_BASE_CAMP_X && y == PLAYER_0_BASE_CAMP_Y) &&
                        !(x == PLAYER_1_BASE_CAMP_X && y == PLAYER_1_BASE_CAMP_Y) &&
                        distance(Pos(x, y), own) <= distance(Pos(x, y), enemy))
                        cells.emplace_back(x, y);
        } else {
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    if (ant_can_walk_to(x, y) &&
                        !(x == PLAYER_0_BASE_CAMP_X && y == PLAYER_0_BASE_CAMP_Y) &&
                        !(x == PLAYER_1_BASE_CAMP_X && y == PLAYER_1_BASE_CAMP_Y))
                        cells.emplace_back(x, y);
        }
        if (cells.empty())
            continue;
        auto cell = cells[random_index(static_cast<int>(cells.size()))];
        ant->teleport_to(cell.first, cell.second);
    }
}

void Game::drift_items() {
    for (int player = 0; player < 2; ++player) {
        for (int index : {ItemType::LightingStorm, ItemType::EMPBlaster}) {
            Item &it = item[player][index];
            if (!it.duration)
                continue;
            std::vector<std::pair<int, int>> cells = {{it.x, it.y}};
            for (int direction = 0; direction < 6; ++direction) {
                int nx = it.x + ant_dx[it.y % 2][direction][0];
                int ny = it.y + ant_dx[it.y % 2][direction][1];
                if (map.is_valid(nx, ny))
                    cells.emplace_back(nx, ny);
            }
            auto cell = cells[random_index(static_cast<int>(cells.size()))];
            it.x = cell.first;
            it.y = cell.second;
        }
    }
}

void Game::init()
{

    round = 0;
    is_end = false;
    winner = -1;
    std::ofstream fout(mini_replay);
    fout.close();
    player0.ant_target_x = PLAYER_1_BASE_CAMP_X;
    player0.ant_target_y = PLAYER_1_BASE_CAMP_Y;
    player1.ant_target_x = PLAYER_0_BASE_CAMP_X;
    player1.ant_target_y = PLAYER_0_BASE_CAMP_Y;

    // read initial info from judger
    from_judger_init judger_init;
    read_from_judger<from_judger_init>(judger_init);
    record_file = judger_init.get_replay();

    json config = judger_init.get_config();
    if (config.contains("random_seed") && config["random_seed"].is_number_unsigned())
    {
        random_seed = config["random_seed"].get<unsigned long long>();
    }
    else if (config.contains("random_seed") &&
             config["random_seed"].is_number_integer())
    {
        long long seed = config["random_seed"].get<long long>();
        random_seed = seed >= 0 ? static_cast<unsigned long long>(seed) : 0ULL;
    }
    else
    {
        std::random_device rd;
        random_seed = rd();
    }
    rng_state = (random_seed ^ RNG_MULTIPLIER) & RNG_MASK;
    map.init_pheromon(random_seed);
    // send config json to judger
    // default config

    if (judger_init.get_player_num() != 2)
    {
        std::cerr << "player_num is not equal to 2\n";
        exit(0);
    }
    // if both players run error, player 1 loses
    for (int i = 0; i < 2; i++)
    {
        if (judger_init.get_AI_state(i) == 1)
        {
            state[i] = AI_state::OK;
            output_to_judger.init_player_state(i, true);
        }
        else if (judger_init.get_AI_state(i) == 2)
        {
            state[i] = AI_state::HUMAN_PLAYER;
            output_to_judger.init_player_state(i, false);
        }
        else
        {
            state[i] = AI_state::INITIAL_ERROR;
            is_end = true;
            winner = (i == 0) ? (1) : (0);
        }
    }
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < ItemType::Count; j++)
        {
            item[i].push_back(Item(0, 0, 0, 0));
        }
    }
    output_to_judger.init_to_player(random_seed, map.get_pheromone());
    base_camp0 = {PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y, 0, 0, 0,
                  INIT_CAMP_HP /*initial hp*/},
    base_camp1 = {PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y, 1, 0, 0,
                  INIT_CAMP_HP /*initial hp*/};
    map.map[PLAYER_0_BASE_CAMP_X][PLAYER_0_BASE_CAMP_Y].base_camp = &base_camp0;
    map.map[PLAYER_1_BASE_CAMP_X][PLAYER_1_BASE_CAMP_Y].base_camp = &base_camp1;
}
void Game::update_items()
{
    drift_items();

    for (int i = 0; i < 2; i++)
        for (auto &it : item[i])
        {
            if (it.duration != 0)
                it.duration -= 1;
            if (it.cd != 0)
                it.cd -= 1;
        }
}
bool Game::is_ended() { return is_end; }

void Game::attack_ants()
{
    prepare_ants_for_attack();
    // super_weapon
    for (int i = 0; i < 2; i++)
    {
        int player = i;
        Item &it = item[player][ItemType::LightingStorm];
        if (it.duration)
        {
            for (auto &ant : ants)
            {
                if (ant.get_player() == !player &&
                    distance(Pos(it.x, it.y), Pos(ant.get_x(), ant.get_y())) <=
                        3)
                {
                    ant.set_hp_true(-100);
                }
            }
        }
    }
    for (auto &tower : defensive_towers)
    {
        if (tower.destroy())
            continue;
        if (tower.is_producer())
            continue;
        // EMP
        Item it = item[!tower.get_player()][ItemType::EMPBlaster];
        if (it.duration &&
            distance(Pos(tower.get_x(), tower.get_y()), Pos(it.x, it.y)) <= 3)
        {
            continue;
        }

        Ant *target = tower.find_attack_target(ants);

        tower.round++;
        if (tower.round >= tower.get_spd() && target != nullptr)
        {
            auto type = tower.get_type();
            tower.set_changed_this_round();
            if (type == TowerType::Mortar || type == TowerType::MortarPlus ||
                type == TowerType::Missile)
            { // AOE
                tower.add_attacked_ants(target->get_id());
                int splash = type == TowerType::Missile ? 2 : 1;
                for (auto &ant : ants) {
                    if (ant.get_player() == tower.get_player())
                        continue;
                    if (distance(Pos(ant.get_x(), ant.get_y()),
                                 Pos(target->get_x(), target->get_y())) <= splash) {
                        damage_ant_by_tower(tower, ant);
                        tower.add_attacked_ants(ant.get_id());
                    }
                }
            }
            else if (type == TowerType::Pulse)
            {
                for (auto &ant : ants) {
                    if (ant.get_player() == tower.get_player())
                        continue;
                    if (distance(Pos(ant.get_x(), ant.get_y()),
                                 Pos(tower.get_x(), tower.get_y())) <= 2) {
                        damage_ant_by_tower(tower, ant);
                        tower.add_attacked_ants(ant.get_id());
                    }
                }
            }
            else if (type == TowerType::Double)
            {
                tower.add_attacked_ants(target->get_id());
                damage_ant_by_tower(tower, *target);
                target = tower.find_attack_target(ants);
                if (target != nullptr)
                {
                    damage_ant_by_tower(tower, *target);
                    tower.add_attacked_ants(target->get_id());
                }
            }
            else if (type == TowerType::QuickPlus)
            {
                tower.add_attacked_ants(target->get_id());
                damage_ant_by_tower(tower, *target);
                target = tower.find_attack_target(ants);
                if (target != nullptr)
                {
                    tower.add_attacked_ants(target->get_id());
                    damage_ant_by_tower(tower, *target);
                }
            }
            else
            { // Single
                tower.add_attacked_ants(target->get_id());
                damage_ant_by_tower(tower, *target);
            }

            tower.round = 0;
        }
        // output.add_tower(tower, TOWER_ATTACK_TYPE, attacked_ant->get_id());
    }
}

void Game::move_ants()
{
    for (auto &ant : ants)
    {
        int move = -1;

        if (ant.get_status() == Ant::Status::Alive) {
            move = choose_ant_move(ant);
        }
        resolve_ant_step(ant, move);
    }
}

void Game::generate_ants()
{
    auto draw_spawn_profile = [this]() {
        double roll = random_float();
        double cumulative = 0.0;
        for (int i = 0; i < 4; ++i) {
            cumulative += SPAWN_BEHAVIOR_PROBS[i];
            if (roll <= cumulative)
                return SPAWN_PROFILES[i];
        }
        return SPAWN_PROFILES[3];
    };

    auto choose_spawn_cell = [this](const DefenseTower &tower) {
        std::vector<std::pair<int, int>> cells;
        Pos enemy = tower.get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                       : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
        double best_score = -1e18;
        std::pair<int, int> best = {tower.get_x(), tower.get_y()};
        for (int direction = 0; direction < 6; ++direction) {
            int nx = tower.get_x() + ant_dx[tower.get_y() % 2][direction][0];
            int ny = tower.get_y() + ant_dx[tower.get_y() % 2][direction][1];
            if (!ant_can_walk_to(nx, ny))
                continue;
            double score = -distance(Pos(nx, ny), enemy);
            score -= crowding_penalty(
                Ant(tower.get_player(), -1, nx, ny,
                    tower.get_player() ? base_camp1.get_ant_level()
                                       : base_camp0.get_ant_level()),
                nx, ny);
            if (score > best_score) {
                best_score = score;
                best = {nx, ny};
            }
        }
        return best;
    };

    auto spawn_from_tower = [&](const DefenseTower &tower, Ant::Kind kind,
                                Ant::Behavior behavior) {
        auto cell = choose_spawn_cell(tower);
        if (!ant_can_walk_to(cell.first, cell.second))
            return;
        int ant_level =
            tower.get_player() ? base_camp1.get_ant_level() : base_camp0.get_ant_level();
        ants.push_back(Ant(tower.get_player(), ant_id, cell.first, cell.second,
                           ant_level, kind));
        ants.back().trail_cells = {Pos(cell.first, cell.second)};
        ants.back().set_behavior(behavior);
        if (kind == Ant::Kind::Combat)
            grant_emergency_evasion(ants.back(), 3, true);
        output.add_ant(ants.back());
        ant_id++;
    };

    if (base_camp0.create_new_ant(round))
    {
        SpawnProfile profile = draw_spawn_profile();
        ants.push_back(Ant(base_camp0.get_player(), ant_id, base_camp0.get_x(),
                           base_camp0.get_y(), base_camp0.get_ant_level(),
                           profile.kind));
        ants.back().set_behavior(profile.behavior);
        if (profile.kind == Ant::Kind::Combat)
            grant_emergency_evasion(ants.back(), 3, true);
        output.add_ant(ants.back());
        ant_id++;
    }
    if (base_camp1.create_new_ant(round))
    {
        SpawnProfile profile = draw_spawn_profile();
        ants.push_back(Ant(base_camp1.get_player(), ant_id, base_camp1.get_x(),
                           base_camp1.get_y(), base_camp1.get_ant_level(),
                           profile.kind));
        ants.back().set_behavior(profile.behavior);
        if (profile.kind == Ant::Kind::Combat)
            grant_emergency_evasion(ants.back(), 3, true);
        output.add_ant(ants.back());
        ant_id++;
    }

    for (auto &tower : defensive_towers) {
        if (tower.destroy() || !tower.is_producer())
            continue;
        Item it = item[!tower.get_player()][ItemType::EMPBlaster];
        if (it.duration &&
            distance(Pos(tower.get_x(), tower.get_y()), Pos(it.x, it.y)) <= 3)
            continue;
        tower.round++;
        if (tower.get_type() == TowerType::ProducerMedic &&
            tower.get_support_interval() > 0 &&
            tower.round % tower.get_support_interval() == 0) {
            Pos enemy = tower.get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                           : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
            int frontline_distance = 1e9;
            for (auto &ant : ants) {
                if (ant.get_player() != tower.get_player())
                    continue;
                auto status = ant.get_status();
                if (status != Ant::Status::Alive && status != Ant::Status::Frozen)
                    continue;
                frontline_distance = std::min(
                    frontline_distance,
                    distance(Pos(ant.get_x(), ant.get_y()), enemy));
            }
            Ant *target = nullptr;
            for (auto &ant : ants) {
                if (ant.get_player() != tower.get_player())
                    continue;
                auto status = ant.get_status();
                if (status != Ant::Status::Alive && status != Ant::Status::Frozen)
                    continue;
                int ant_distance = distance(Pos(ant.get_x(), ant.get_y()), enemy);
                if (ant_distance > frontline_distance + 1)
                    continue;
                if (target == nullptr ||
                    (target->get_kind() != Ant::Kind::Combat &&
                     ant.get_kind() == Ant::Kind::Combat) ||
                    (target->get_kind() == ant.get_kind() &&
                     (ant.get_hp() < target->get_hp() ||
                      (ant.get_hp() == target->get_hp() &&
                       (ant_distance <
                            distance(Pos(target->get_x(), target->get_y()), enemy) ||
                        (ant_distance ==
                             distance(Pos(target->get_x(), target->get_y()), enemy) &&
                         ant.get_id() < target->get_id())))))) {
                    target = &ant;
                }
            }
            if (target != nullptr) {
                target->set_hp_true(target->get_hp_limit() - target->get_hp());
                target->add_evasion(1, true);
            }
        }
        if (tower.round < tower.get_spawn_interval())
            continue;
        SpawnProfile profile = draw_spawn_profile();
        spawn_from_tower(tower, profile.kind, profile.behavior);
        if (tower.get_type() == TowerType::ProducerSiege &&
            random_float() <= tower.get_siege_spawn_chance()) {
            spawn_from_tower(tower, Ant::Kind::Combat, Ant::Behavior::Default);
        }
        tower.round = 0;
    }
}

// if after one ant moves, base_camp of a player < 0, then return true
bool Game::manage_ants()
{

    /* save output, remove fail ant */
    for (auto ant_it = ants.begin(); ant_it != ants.end();)
    {
        output.add_ant(*ant_it);

        if (ant_it->get_status() == Ant::Status::Success)
        {
            if (ant_it->get_player())
            {
                base_camp0.set_hp(-1);
                player1.coin.income_ant_arrive();
            }
            else
            {
                base_camp1.set_hp(-1);
                player0.coin.income_ant_arrive();
            }
            ant_it = ants.erase(ant_it);
            if (judge_base_camp())
            {
                return false;
            }
        }
        else if (ant_it->get_status() == Ant::Status::Fail)
        {
            if (ant_it->get_player() == 1)
            {
                player0.coin.income_ant_kill(*ant_it);
                player0.opponent_killed_ant++;
            }
            else
            {
                player1.coin.income_ant_kill(*ant_it);
                player1.opponent_killed_ant++;
            }
            ant_it = ants.erase(ant_it);
        }
        else
        {
            ++ant_it;
        }
    }
    /* remove old*/
    for (auto ant_it = ants.begin(); ant_it != ants.end();) {
        if (ant_it->get_status() == Ant::Status::TooOld) {
            ant_it = ants.erase(ant_it);
        }
        else
        {
            ant_it++;
        }
    }
    return true;
}

void Game::increase_ant_age() {
    for (auto &ant : ants) {
        ant.increase_age();
        ant.increase_behavior_rounds();
        if (ant.get_behavior() == Ant::Behavior::Randomized &&
            ant.behavior_rounds >= RANDOM_ANT_DECAY_TURNS) {
            ant.set_behavior(Ant::Behavior::Default, true, -1, true);
        } else {
            if (ant.get_behavior() == Ant::Behavior::Bewitched &&
                ant.reached_target()) {
                ant.set_behavior(Ant::Behavior::Default, true, -1, true);
            } else if (ant.behavior_expiry > 0) {
                ant.behavior_expiry--;
                if (ant.get_behavior() != Ant::Behavior::Default &&
                    ant.get_behavior() != Ant::Behavior::Randomized &&
                    ant.behavior_expiry <= 0) {
                    ant.set_behavior(Ant::Behavior::Default, true, -1, true);
                }
            }
        }
    }
}

// when game ends, return true
bool Game::judge_base_camp()
{
    if (base_camp0.get_hp() <= 0 && base_camp1.get_hp() <= 0)
    {
        // player 0 wins
        is_end = 1;
        winner = 0;
        return true;
    }
    else if (base_camp1.get_hp() <= 0)
    {
        is_end = 1;
        winner = 0;
        return true;
    }
    else if (base_camp0.get_hp() <= 0)
    {
        is_end = 1;
        winner = 1;
        return true;
    }
    else
    {
        return false;
    }
}

void Game::judge_winner()
{
    // judge base_camp
    if (base_camp0.get_hp() < base_camp1.get_hp())
    {
        winner = 1;
        return;
    }
    else if (base_camp0.get_hp() > base_camp1.get_hp())
    {
        winner = 0;
        return;
    }
    else
    {
        // judge kiiled ants
        if (player0.opponent_killed_ant > player1.opponent_killed_ant)
        {
            winner = 0;
            return;
        }
        else if (player1.opponent_killed_ant > player0.opponent_killed_ant)
        {
            winner = 1;
            return;
        }
        else
        {
            // judge super weapons usage
            if (player0.super_weapons_usage < player1.super_weapons_usage)
            {
                winner = 0;
                return;
            }
            else if (player0.super_weapons_usage >
                     player1.super_weapons_usage)
            {
                winner = 1;
                return;
            }
            else
            {
                // judge AI_total_time
                if (player0.AI_total_time < player1.AI_total_time)
                {
                    winner = 0;
                    return;
                }
                else if (player0.AI_total_time > player1.AI_total_time)
                {
                    winner = 1;
                    return;
                }
                else
                {
                    // player 0 wins
                    winner = 0;
                    return;
                }
            }
        }
    }
}

void Game::update_coin()
{
    std::tuple<bool, int> coin0 = player0.coin.basic_income_and_penalty();
    std::tuple<bool, int> coin1 = player1.coin.basic_income_and_penalty();
    if (std::get<0>(coin0))
        player0.coin.set_coin(std::get<1>(coin0));
    else
        base_camp0.set_hp(std::get<1>(coin0));
    if (std::get<0>(coin1))
        player1.coin.set_coin(std::get<1>(coin1));
    else
        base_camp1.set_hp(std::get<1>(coin1));
}

// when game ends, return false
bool Game::next_round()
{
    // std::ofstream fout;
    // out.open("test_2.out");

    attack_ants();
    // fout << "atk "<< std::endl;
    move_ants();
    teleport_ants();
    // fout << "mov "<< std::endl;
    update_pheromone();
    // fout << "upp "<< std::endl;
    bool should_continue = manage_ants();
    // fout << "mng "<< std::endl;
    if (!should_continue)
    {
        round++;
        return false;
    }
    generate_ants();
    increase_ant_age();
    // fout << "gen "<< std::endl;
    update_coin();
    update_items();
    round++;
    if (round == MAX_ROUND)
    {
        is_end = 1;
        judge_winner();
        return false;
    }

    if (judge_base_camp())
    {
        return false;
    }
    return true;
}
void Game::update_pheromone()
{
    map.next_round();

    /* update pheromone*/
    for (auto ant = ants.begin(); ant != ants.end(); ant++)
    {
        map.update_pheromone(&*ant);
    }
}

bool Game::apply_operation(const std::vector<Operation> &op_list, int player,
                           std::string &err_msg)
{
    op[player] = op_list;
    bool camp_upgraded_flag = false;
    std::vector<int> used_tower;
    for (auto &op : op_list)
    {
        int x = op.get_pos_x();
        int y = op.get_pos_y();
        switch (op.get_operation_type())
        {
        case Operation::Type::TowerBuild:
        {
            /*if (!map.is_empty(x, y, player)) { // position judge
                err_msg = "TowerBuild: position is not empty";
                return false;
            }*/
            if (x >= MAP_SIZE || y >= MAP_SIZE || x < 0 || y < 0)
            {
                char msg[100];
                sprintf(msg, "TowerBuild: position out of range (at %d, %d)", x, y);
                err_msg = msg;
                return false;
            }
            if (map.map[x][y].base_camp != nullptr)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: attempt to build a tower (at %d, %d), in which there is already a camp. (player id = %d)",
                        x, y, map.map[x][y].player);
                err_msg = msg;
                return false;
            }
            if (map.map[x][y].tower != nullptr)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: attempt to build a tower (at %d, %d), in which there is already a tower. (player id = %d)",
                        x, y, map.map[x][y].player);
                err_msg = msg;
                return false;
            }
            if (map.map[x][y].player != player)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: Build a tower at position (%d, %d), its player is %d, request player = %d", x, y, map.map[x][y].player, player);
                err_msg = msg;
                return false;
            }
            if (player == 1 &&
                !player1.coin.isEnough_tower_build())
            { // not enough money
                err_msg = "TowerBuild: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_tower_build())
            {
                err_msg = "TowerBuild: P0 not enough money";
                return false;
            }
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration && distance(Pos(x, y), Pos(it.x, it.y)) <= 3)
            {
                err_msg = "TowerBuild: EMPBlaster is active";
                return false;
            }

            if (player == 1)
                player1.coin.cost_tower_build();
            else
                player0.coin.cost_tower_build();

            used_tower.push_back(tower_id);
            defensive_towers.push_back(DefenseTower{x, y, player, tower_id, 0});
            DefenseTower &new_tower = defensive_towers.back();
            map.build(&new_tower);
            new_tower.set_changed_this_round();
            // output.add_tower(new_tower, TOWER_BUILD_TYPE);
            tower_id++;
            break;
        }
        case Operation::Type::TowerUpgrade:
        {
            int id = op.get_id();
            if (id < 0 || id >= (int)defensive_towers.size() ||
                defensive_towers[id].destroy()|| defensive_towers[id].get_player() != player)
            {
                err_msg = "TowerUpgrade: Invalid Tower id";
                return false;
            }
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
            {
                err_msg = "TowerUpgrade: Tower has been used";
                return false;
            }
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration && distance(Pos(x, y), Pos(it.x, it.y)) <= 3)
            {
                err_msg = "TowerUpgrade: EMPBlaster is active";
                return false;
            }

            DefenseTower &tower = defensive_towers[id];
            if (tower.get_level() ==
                TOWER_MAX_LEVEL)
            { // have reached max level
                err_msg = "TowerUpgrade: Tower has reached max level";
                return false;
            }
            if (player == 1 && !player1.coin.isEnough_tower_upgrade(
                                   tower))
            { // not enough money
                err_msg = "TowerUpgrade: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_tower_upgrade(tower))
            {
                err_msg = "TowerUpgrade: P0 not enough money";
                return false;
            }
            if (!tower.upgrade_type_check(op.get_args()))
            {
                err_msg = "TowerUpgrade: Invalid upgrade type";
                return false;
            }

            used_tower.push_back(id);
            if (player == 1) // must cost coin first!!
                player1.coin.cost_tower_upgrade(tower);
            else
                player0.coin.cost_tower_upgrade(tower);

            tower.upgrade(TowerType(op.get_args()));
            tower.set_changed_this_round();
            // output.add_tower(tower, op.get_args());
            break;
        }
        case Operation::Type::TowerDestroy:
        {
            int id = op.get_id();
            if (id < 0 || id >= (int)defensive_towers.size() ||
                defensive_towers[id].destroy() || defensive_towers[id].get_player() != player)
            {
                err_msg = "TowerDestroy: Invalid Tower id";
                return false;
            }
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
            {
                err_msg = "TowerDestroy: Tower has been used";
                return false;
            }
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration && distance(Pos(x, y), Pos(it.x, it.y)) <= 3)
            {
                err_msg = "TowerDestroy: EMPBlaster is active";
                return false;
            }
            DefenseTower *defensive_tower = &defensive_towers[id];
            if (player == 1)
                player1.coin.income_tower_destroy(defensive_tower->get_level());
            else
                player0.coin.income_tower_destroy(defensive_tower->get_level());

            if (defensive_tower->get_type() == TowerType::Basic)
            {
                map.destroy(defensive_tower->get_x(), defensive_tower->get_y());
                output.add_tower(*defensive_tower, TOWER_DESTROY_TYPE,
                                 defensive_tower->get_attack());
                defensive_tower->set_destroy();
            }
            else
            {
                TowerType new_type = defensive_tower->tower_downgrade_type();
                defensive_tower->downgrade(new_type);
                defensive_tower->set_changed_this_round();
                // output.add_tower(*defensive_tower, new_type);
            }
            used_tower.push_back(id);
            break;
        }
        case Operation::Type::LightingStorm:
        {
            if (x < 0 || x >= MAP_SIZE || y < 0 ||
                y >= MAP_SIZE)
            { // position judge
                err_msg = "LightingStorm: invaid position";
                return false;
            }
            ItemType it = ItemType::LightingStorm;
            if (item[player][it].cd)
            {
                err_msg = "LightingStorm: in CD";
                return false;
            }
            if (player == 1 &&
                !player1.coin.isEnough_item_applied(it))
            { // not enough money
                err_msg = "LightingStorm: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_item_applied(it))
            {
                err_msg = "LightingStorm: P0 not enough money";
                return false;
            }

            if (player == 1)
                player1.coin.cost_item(it);
            else
                player0.coin.cost_item(it);

            if (player == 0)
                player0.super_weapons_usage++;
            else
                player1.super_weapons_usage++;
            item[player][it] = Item(it, x, y);

            break;
        }
        case Operation::Type::EMPBlaster:
        {
            if (x < 0 || x >= MAP_SIZE || y < 0 ||
                y >= MAP_SIZE)
            { // position judge
                err_msg = "EMPBlaster: invaid position";
                return false;
            }
            ItemType it = ItemType::EMPBlaster;
            if (item[player][it].cd)
            {
                err_msg = "EMPBlaster: in CD";
                return false;
            }
            if (player == 1 &&
                !player1.coin.isEnough_item_applied(it))
            { // not enough money
                err_msg = "EMPBlaster: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_item_applied(it))
            {
                err_msg = "EMPBlaster: P0 not enough money";
                return false;
            }
            if (player == 1)
                player1.coin.cost_item(it);
            else
                player0.coin.cost_item(it);

            if (player == 0)
                player0.super_weapons_usage++;
            else
                player1.super_weapons_usage++;
            item[player][it] = Item(it, x, y);
            break;
        }
        case Operation::Type::Deflectors:
        {
            if (x < 0 || x >= MAP_SIZE || y < 0 ||
                y >= MAP_SIZE)
            { // position judge
                err_msg = "Deflectors: invaid position";
                return false;
            }
            ItemType it = ItemType::Deflectors;
            if (item[player][it].cd)
            {
                err_msg = "Deflectors: in CD";
                return false;
            }
            if (player == 1 &&
                !player1.coin.isEnough_item_applied(it))
            { // not enough money
                err_msg = "Deflectors: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_item_applied(it))
            {
                err_msg = "Deflectors: P0 not enough money";
                return false;
            }
            if (player == 1)
                player1.coin.cost_item(it);
            else
                player0.coin.cost_item(it);

            item[player][it] = Item(it, x, y);

            if (player == 0)
                player0.super_weapons_usage++;
            else
                player1.super_weapons_usage++;

            break;
        }
        case Operation::Type::EmergencyEvasion:
        {
            if (x < 0 || x >= MAP_SIZE || y < 0 ||
                y >= MAP_SIZE)
            { // position judge
                err_msg = "EmergencyEvasion: invaid position";
                return false;
            }
            ItemType it = ItemType::EmergencyEvasion;
            if (item[player][it].cd)
            {
                err_msg = "EmergencyEvasion: in CD";
                return false;
            }
            if (player == 1 &&
                !player1.coin.isEnough_item_applied(it))
            { // not enough money
                err_msg = "EmergencyEvasion: P1 not enough money";
                return false;
            }
            if (player == 0 && !player0.coin.isEnough_item_applied(it))
            {
                err_msg = "EmergencyEvasion: P0 not enough money";
                return false;
            }
            if (player == 1)
                player1.coin.cost_item(it);
            else
                player0.coin.cost_item(it);

            if (player == 0)
                player0.super_weapons_usage++;
            else
                player1.super_weapons_usage++;
            for (auto &ant : ants)
            {
                if (ant.get_player() == player &&
                    distance(Pos(x, y), Pos(ant.get_x(), ant.get_y())) <= 3)
                {
                    grant_emergency_evasion(ant, 2, true);
                }
            }
            item[player][it] = Item(it, x, y);

            break;
        }

        case Operation::Type::BarrackUpgrade:
        {
            Headquarter &base_camp = player ? base_camp1 : base_camp0;
            if (camp_upgraded_flag)
            {
                err_msg = "BarrackUpgrade: already upgraded this tern";
                return false;
            }
            int level = base_camp.get_cd_level();
            if (level == 2)
            {
                err_msg = "BarrackUpgrade: already max level";
                return false;
            }
            if (player == 1 && !player1.coin.isEnough_base_camp_upgrade(
                                   level))
            { // not enough money
                err_msg = "BarrackUpgrade: P1 not enough money";
                return false;
            }
            if (player == 0 &&
                !player0.coin.isEnough_base_camp_upgrade(level))
            {
                err_msg = "BarrackUpgrade: P0 not enough money";
                return false;
            }
            camp_upgraded_flag = true;
            if (player == 1)
                player1.coin.cost_base_camp_upgrade(level);
            else
                player0.coin.cost_base_camp_upgrade(level);
            base_camp.barrack_upgrade();
            break;
        }
        case Operation::Type::AntUpgrade:
        {
            Headquarter &base_camp = player ? base_camp1 : base_camp0;
            if (camp_upgraded_flag)
            {
                err_msg = "BarrackUpgrade: already upgraded this tern";
                return false;
            }
            int level = base_camp.get_ant_level();
            if (level == 2)
            {
                err_msg = "AntUpgrade: already max level";
                return false;
            }
            if (player == 1 && !player1.coin.isEnough_base_camp_upgrade(
                                   level))
            { // not enough money
                err_msg = "AntUpgrade: P1 not enough money";
                return false;
            }
            if (player == 0 &&
                !player0.coin.isEnough_base_camp_upgrade(level))
            {
                err_msg = "AntUpgrade: P0 not enough money";
                return false;
            }
            camp_upgraded_flag = true;
            if (player == 1)
                player1.coin.cost_base_camp_upgrade(level);
            else
                player0.coin.cost_base_camp_upgrade(level);

            base_camp.ant_upgrade();
            break;
        }

        // case Operation::Type::PutAnt:
        // {
        //     if (!map.is_valid(x, y)) // position judge
        //         return false;

        //     ants.push_back(Ant(player, ant_id, x, y, 5));
        //     ant_id++;
        //     break;
        // }
        // case Operation::Type::DeleteAnt:
        // {
        //     int id = op.get_id();
        //     auto ant =
        //         std::find_if(ants.begin(), ants.end(), [id](const Ant &ant)
        //                      { return id == ant.get_id(); });
        //     if (ant == ants.end())
        //     {
        //         return false;
        //     }

        //     ants.erase(ant);
        //     break;
        // }
        // case Operation::Type::MaxCoin:
        //     if (player == 1)
        //         player1.coin.set_coin(100000);
        //     else
        //         player0.coin.set_coin(100000);
        //     break;
        default:
            return false;
        }
    }
    return true;
}

// void Game::dump_mini_replay(const std::string &filename) {
//     std::ofstream fout(filename, std::ios_base::app);
//     fout << round <<std::endl;
//     fout << player0.coin.get_coin() << " " << player1.coin.get_coin() <<
//     std::endl; fout << base_camp0.get_hp() << " " << base_camp1.get_hp() <<
//     std::endl; fout << barracks.size() << std::endl; for(auto barrack :
//     barracks) {
//         if(barrack.destroy()) continue;
//         fout << barrack.get_id() << " " << barrack.get_x() << " " <<
//         barrack.get_y() << " " << barrack.get_player() << std::endl;
//     }
//     fout << ants.size() << std::endl;
//     for(auto ant : ants) {
//         fout << ant.get_id() << " " << ant.get_x() << " " << ant.get_y() << "
//         " <<
//             ant.get_player() << " " << ant.get_hp() << " " <<
//             ant.get_status() << std::endl;
//     }
//     fout << defensive_towers.size() << std::endl;
//     for(auto tower : defensive_towers) {
//         if(tower.destroy()) continue;
//         fout << tower.get_id() << " " << tower.get_x() << " " <<
//         tower.get_y() << " " <<
//             tower.get_player() << " " << tower.get_type() << std::endl;
//     }
//     // items
//     std::vector<Item> exist_items;
//     for(auto item : items) {
//         if(item.get_state(round) != ItemState::Exist) continue;
//         exist_items.push_back(item);
//     }
//     fout << exist_items.size() << std::endl;
//     for(auto item : exist_items) {
//         fout << item.get_id() << " " << item.get_pos().x << " " <<
//         item.get_pos().y << " " << item.get_type() << std::endl;
//     }
//     // applied items
//     fout << buff_list.size() << std::endl;
//     for(auto buff : buff_list) {
//         fout << std::get<3>(buff) << " " << std::get<1>(buff) << " " <<
//         std::get<0>(buff) << std::endl;
//     }
//     fout.close();
// }

void Game::dump_round_state(/* const std::string &filename */)
{
    // state info
    for (auto &tower : defensive_towers)
    {
        if (!tower.is_changed())
            continue;
        if (tower.destroy())
        {
            output.add_tower(tower, TOWER_DESTROY_TYPE, tower.get_attack());
            tower.set_unchanged_before_another_round();
            continue;
        }
        if (tower.is_changed())
        {
            output.add_tower(tower, tower.get_type(), tower.get_attack());
            tower.set_unchanged_before_another_round();
        }
    }
    output.add_camps(base_camp0, base_camp1);
    output.add_coins(player0.coin, player1.coin);
    output.add_pheromone(map.get_pheromone());
    output.add_winner(winner, "");
    if (!err_msg.empty())
        output.add_error(err_msg);

    // replay info
    output.add_operation(op);
    if (round == 1) {
        output.save_seed(random_seed);
    }
    output.save_data();
    // output.dump_cur(filename);
    output_to_judger.set_json_to_web_player(output.get_cur());
    output.update_cur(defensive_towers);

    // mini replay info
    // dump_mini_replay(mini_replay);

    // change cur to another new json so that it can send needed message to ai
    output_to_judger.send_info_to_judger(output.get_cur(), round);
    output.next_round();
}

void Game::dump_last_round(
    /* const std::string &filename */ const std::string &msg)
{
    for (auto tower : defensive_towers)
    {
        if (tower.is_changed() && tower.destroy())
        {
            output.add_tower(tower, TOWER_DESTROY_TYPE, tower.get_attack());
        }
        else if (tower.is_changed())
        {
            output.add_tower(tower, tower.get_type(), tower.get_attack());
        }
    }
    output.add_camps(base_camp0, base_camp1);
    output.add_coins(player0.coin, player1.coin);
    output.add_pheromone(map.get_pheromone());

    output.add_winner(winner, msg);
    output.add_operation(op);
    output.add_error(err_msg);
    output.save_data();
    // output.dump_cur(filename);
    output_to_judger.set_json_to_web_player(output.get_cur());

    output.update_cur(defensive_towers);
    // dump_mini_replay(mini_replay);
    output_to_judger.send_info_to_judger(output.get_cur(), round);
}

void Game::dump_result(const std::string &filename)
{
    output.dump_all(filename);
}

Game::Game() {}

Game::~Game()
{
    // TO DO (important?)
}

bool Game::round_read_from_judger(int player)
{
    // player 0 & player 1

    read_from_judger<from_judger_round>(judger_round_info);

    std::string content = judger_round_info.get_content();

    if (judger_round_info.get_player() == -1)
    {
        json error;
        try
        {
            error = json::parse(content);
            int AI_ID = error["player"].get<int>();
            switch (error["error"].get<int>())
            {
            case 0:
            {
                state[AI_ID] = AI_state::RUN_ERROR;
                break;
            }
            case 1:
            {
                state[AI_ID] = AI_state::TIMEOUT_ERROR;
                break;
            }
            case 2:
            {
                state[AI_ID] = AI_state::OUTPUT_LIMIT;
                break;
            }
            default:
            {
                break;
            }
            }
            is_end = true;
            winner = (AI_ID == 0) ? (1) : (0);
            return false;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Information of ai's error is not json\n";
            std::cerr << content << '\n';
            winner = -1;
            return false;
        }
    }
    else
    {
        judger_round_info.transfer_op(output_to_judger.if_ai(player));

        if (judger_round_info.get_player() != player)
        {
            is_end = true;
            winner = player;
            return false;
        }

        int another_player = 1 - player;

        std::vector<Operation> op_list = judger_round_info.get_op_list();
        if (!apply_operation(op_list, player, err_msg))
        {
            set_AI_state_IO(player);
            return false;
        }

        judger_round_info.send_operation(
            output_to_judger.if_ai(another_player));

        // update AI_total_time
        if (player == 0)
        {
            player0.AI_total_time += judger_round_info.get_time();
        }
        else
        {
            player1.AI_total_time += judger_round_info.get_time();
        }

        return true;
    }
}

void Game::request_end_state()
{
    json end_request = {{"action", "request_end_state"}};
    output_info(-1, end_request);
}

void Game::receive_end_state()
{
    end_from_judger end_state;
    read_from_judger<end_from_judger>(end_state);

    // use this to judge scores
    // TO DO
}

void Game::send_end_info()
{
    // scores of players, TO DO
    int score[2] = {0, 0};
    if (winner == 0)
    {
        score[0] = 1;
        score[1] = 0;
    }
    else if (winner == 1)
    {
        score[0] = 0;
        score[1] = 1;
    }
    json end_info_json = {
        {"0", score[0]},
        {"1", score[1]},
    };
    std::string end_info = end_info_json.dump();
    std::string end_state = "[";
    // state of player 0 & player 1
    std::string AI_state_info[2] = {"OK", "OK"};
    for (int i = 0; i <= 1; i++)
    {
        switch (state[i])
        {
        case AI_state::OK:
        {
            AI_state_info[i] = "OK";
            break;
        }
        case AI_state::INITIAL_ERROR:
        {
            // AI_state_info[i] = "INITIAL_ERROR";
            AI_state_info[i] = "RE";
            break;
        }
        case AI_state::RUN_ERROR:
        {
            // AI_state_info[i] = "RUN_ERROR";
            AI_state_info[i] = "RE";
            break;
        }
        case AI_state::TIMEOUT_ERROR:
        {
            // AI_state_info[i] = "TIMEOUT_ERROR";
            AI_state_info[i] = "TLE";
            break;
        }
        case AI_state::OUTPUT_LIMIT:
        {
            // AI_state_info[i] = "OUTPUT_LIMIT";
            AI_state_info[i] = "OLE";
            break;
        }
        case AI_state::ILLEGAL_OPERATION:
        {
            // AI_state_info[i] = "ILLEGAL_OPERATION";
            AI_state_info[i] = "IA";
            break;
        }
        case AI_state::HUMAN_PLAYER:
        {
            // AI_state_info[i] = "HUMAN_PLAYER";
            AI_state_info[i] = "OK";
            break;
        }

        default:
        {
            break;
        }
        }
    }
    end_state =
        end_state + "\"" + AI_state_info[0] /*end state of player 0*/ + "\", ";
    end_state =
        end_state + "\"" + AI_state_info[1] /*end state of player 1*/ + "\"]";
    json end_message = {
        {"state", -1}, {"end_info", end_info}, {"end_state", end_state}};

    dump_last_round(/* "output.json" */ end_state);

    dump_result(get_record_file());
    output_info(-1, end_message);
}

std::string Game::get_record_file() { return record_file; }

void Game::set_AI_state_IO(int player)
{
    state[player] = AI_state::ILLEGAL_OPERATION;
    is_end = true;
    winner = (player == 0) ? (1) : (0);
}

template <typename T>
void Game::read_from_judger(T &des)
{
    std::uint32_t length = 0;
    for (int i = 0; i < 4; ++i) {
        int byte = getchar();
        if (byte == EOF) {
            std::cerr << "read from judger error\n";
            std::cerr << "unexpected EOF while reading packet length\n";
            exit(0);
        }
        length = (length << 8) + static_cast<unsigned char>(byte);
    }
    if (length > MAX_JUDGER_PACKET_SIZE) {
        std::cerr << "read from judger error\n";
        std::cerr << "packet too large: " << length << '\n';
        exit(0);
    }

    std::string in(length, '\0');
    for (std::uint32_t i = 0; i < length; ++i)
    {
        int byte = getchar();
        if (byte == EOF) {
            std::cerr << "read from judger error\n";
            std::cerr << "unexpected EOF while reading packet body\n";
            exit(0);
        }
        in[i] = static_cast<char>(byte);
    }
    json judger_json;

    if (!try_parse_json_payload(in, judger_json))
    {
        std::cerr << "read from judger error\n";
        std::cerr << in << '\n';
        exit(0);
    }
    des = judger_json;
}

void Game::listen(int player) { output_to_judger.listen_player(player); }
