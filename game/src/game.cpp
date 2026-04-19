#include "../include/game.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
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
constexpr double WORKER_RISK_FIELD_DISTANCE_DECAY = 0.9;
constexpr double COMBAT_RISK_FIELD_DISTANCE_DECAY = 0.7;
constexpr double DAMAGE_FIELD_HP_REFERENCE = 25.0;
constexpr int RANDOM_ANT_DECAY_TURNS = 5;
constexpr int ANT_TELEPORT_INTERVAL = 10;
constexpr double ANT_TELEPORT_RATIO = 0.1;
constexpr int LIGHTNING_STORM_ANT_DAMAGE = 20;
constexpr int LIGHTNING_STORM_TOWER_DAMAGE = 3;
constexpr int LIGHTNING_STORM_TOWER_INTERVAL = 5;
constexpr double DEFLECTOR_PATH_ATTRACTION = 1.0;
constexpr double EMERGENCY_EVASION_PATH_ATTRACTION = 1.35;
constexpr double STALL_MOVE_PENALTY = 0.35;
constexpr double RETREAT_MOVE_PENALTY = 0.8;
constexpr double TARGET_PULL_DISTANCE_SCALE = 0.18;
constexpr int COMBAT_SELF_DESTRUCT_DAMAGE = 10;
constexpr int COMBAT_SELF_DESTRUCT_RANGE = 1;
constexpr double COMBAT_SELF_DESTRUCT_PULL_BONUS = 3.0;
constexpr double COMBAT_TOWER_TARGET_BONUS = 8.0;
constexpr double COMBAT_TOWER_APPROACH_PULL_BASE = 8.0;
constexpr double WORKER_TOWER_TARGET_BONUS = 2.75;
constexpr double WORKER_PATH_DAMAGE_WEIGHT = 0.20;
constexpr double WORKER_PATH_CONTROL_WEIGHT = 1.80;
constexpr double WORKER_PATH_TRAFFIC_WEIGHT = 0.75;
constexpr double WORKER_PATH_EFFECT_WEIGHT = 0.35;
constexpr double WORKER_RESERVATION_WEIGHT = 1.40;
constexpr double WORKER_TOWER_CLAIM_WEIGHT = 1.00;
constexpr double WORKER_BLOCKED_ATTACK_BONUS = 6.00;
constexpr double WORKER_ROUTE_IMPROVEMENT_EPS = 0.50;
constexpr double COMBAT_PATH_DAMAGE_WEIGHT = 0.08;
constexpr double COMBAT_PATH_CONTROL_WEIGHT = 0.45;
constexpr double COMBAT_PATH_TRAFFIC_WEIGHT = 0.25;
constexpr double COMBAT_PATH_EFFECT_WEIGHT = 0.20;
constexpr double COMBAT_RESERVATION_WEIGHT = 0.45;
constexpr double COMBAT_TOWER_CLAIM_WEIGHT = 0.85;
constexpr double COMBAT_TRAVEL_COST_WEIGHT = 0.90;
constexpr double ATTACK_FINISH_BONUS = 3.00;
constexpr double SURPLUS_HP_VALUE_WEIGHT = 0.15;
constexpr double ENHANCED_COMBAT_ATTACK_EXECUTION_BONUS = 1.50;
constexpr double WORKER_REROUTE_ATTACK_PENALTY_WEIGHT = 1.0;
constexpr double MIN_PATH_STEP_COST = 0.15;
constexpr double SPAWN_BEHAVIOR_PROBS[4] = {0.4, 0.35, 0.10, 0.15};
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

Game::MovementPolicy parse_movement_policy(const json &config) {
    if (!config.contains("movement_policy") || !config["movement_policy"].is_string())
        return Game::MovementPolicy::Enhanced;
    const std::string policy = config["movement_policy"].get<std::string>();
    if (policy == "legacy")
        return Game::MovementPolicy::Legacy;
    return Game::MovementPolicy::Enhanced;
}

bool parse_cold_handle_rule_illegal(const json &config) {
    if (!config.contains("cold_handle_rule_illegal") ||
        !config["cold_handle_rule_illegal"].is_boolean()) {
        return false;
    }
    return config["cold_handle_rule_illegal"].get<bool>();
}

bool is_base_upgrade_operation(Operation::Type type) {
    return type == Operation::Type::BarrackUpgrade ||
           type == Operation::Type::AntUpgrade;
}

void append_error_message(std::string &target, const std::string &message) {
    if (message.empty())
        return;
    if (!target.empty())
        target += " | ";
    target += message;
}

void append_illegal_summary(std::string &target, int player,
                            const std::vector<std::string> &messages) {
    if (messages.empty())
        return;
    std::string summary = "P" + std::to_string(player) + " ignored " +
                          std::to_string(messages.size()) + " illegal ops";
    if (!messages.front().empty())
        summary += ": " + messages.front();
    if (messages.size() > 1)
        summary += " ...";
    append_error_message(target, summary);
}

bool lightning_storm_tower_strike_turn(int remaining_duration) {
    int active_turn = get_item_time(ItemType::LightingStorm) - remaining_duration + 1;
    return active_turn > 0 && active_turn % LIGHTNING_STORM_TOWER_INTERVAL == 0;
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

std::vector<std::tuple<int, int, int>>
Game::legal_move_candidates(const Ant &ant) const {
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
    return candidates;
}

int Game::choose_random_legal_move(const Ant &ant) {
    std::vector<std::tuple<int, int, int>> candidates = legal_move_candidates(ant);
    if (candidates.empty())
        return Ant::NoMove;
    return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
}

void Game::mark_risk_fields_dirty() {
    risk_fields_dirty = true;
    invalidate_enhanced_move_cache();
}

void Game::refresh_static_risk_fields() {
    if (!risk_fields_dirty)
        return;
    for (int player = 0; player < 2; ++player)
        for (int x = 0; x < MAP_SIZE; ++x)
            for (int y = 0; y < MAP_SIZE; ++y) {
                damage_risk_field[player][x][y] = 0.0;
                control_risk_field[player][x][y] = 0.0;
                effect_pull_field[player][x][y] = 0.0;
            }

    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.is_producer())
            continue;
        int threatened_player = !tower.get_player();
        double damage_value =
            static_cast<double>(tower.get_damage()) / DAMAGE_FIELD_HP_REFERENCE;
        double control_value = 0.0;
        switch (tower.get_type()) {
        case TowerType::Ice:
            control_value = 1.0;
            break;
        case TowerType::Bewitch:
            control_value = 1.3;
            break;
        case TowerType::Pulse:
            control_value = 0.7;
            break;
        default:
            break;
        }
        for (int x = 0; x < MAP_SIZE; ++x)
            for (int y = 0; y < MAP_SIZE; ++y) {
                if (!ant_can_walk_to(x, y))
                    continue;
                if (distance(Pos(x, y), Pos(tower.get_x(), tower.get_y())) >
                    tower.get_range())
                    continue;
                damage_risk_field[threatened_player][x][y] += damage_value;
                if (control_value > 0.0)
                    control_risk_field[threatened_player][x][y] += control_value;
            }
    }
    const double storm_damage =
        static_cast<double>(LIGHTNING_STORM_ANT_DAMAGE) /
        DAMAGE_FIELD_HP_REFERENCE;
    for (int player = 0; player < 2; ++player) {
        Item &storm = item[player][ItemType::LightingStorm];
        if (storm.duration) {
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    if (ant_can_walk_to(x, y) &&
                        distance(Pos(x, y), Pos(storm.x, storm.y)) <= 3)
                        damage_risk_field[!player][x][y] += storm_damage;
        }
        Item &deflect = item[player][ItemType::Deflectors];
        if (deflect.duration) {
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    if (ant_can_walk_to(x, y) &&
                        distance(Pos(x, y), Pos(deflect.x, deflect.y)) <= 3)
                        effect_pull_field[player][x][y] +=
                            DEFLECTOR_PATH_ATTRACTION;
        }
        Item &evasion = item[player][ItemType::EmergencyEvasion];
        if (evasion.duration) {
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    if (ant_can_walk_to(x, y) &&
                        distance(Pos(x, y), Pos(evasion.x, evasion.y)) <= 3)
                        effect_pull_field[player][x][y] +=
                            EMERGENCY_EVASION_PATH_ATTRACTION;
        }
    }
    risk_fields_dirty = false;
}

void Game::invalidate_enhanced_move_cache() {
    enhanced_move_cache_dirty = true;
    if (!enhanced_move_phase_active) {
        enhanced_move_cells.clear();
        enhanced_move_tower_targets.clear();
    }
}

void Game::begin_move_phase() {
    if (movement_policy != MovementPolicy::Enhanced)
        return;
    enhanced_move_phase_active = true;
    prepare_enhanced_move_cache(true);
}

void Game::end_move_phase() {
    if (movement_policy != MovementPolicy::Enhanced)
        return;
    enhanced_move_phase_active = false;
    enhanced_move_cells.clear();
    enhanced_move_tower_targets.clear();
    invalidate_enhanced_move_cache();
}

double Game::cell_damage_hp(int player, int x, int y) const {
    return damage_risk_field[player][x][y] * DAMAGE_FIELD_HP_REFERENCE;
}

void Game::compute_enhanced_traffic_field() {
    for (int player = 0; player < 2; ++player)
        for (int x = 0; x < MAP_SIZE; ++x)
            for (int y = 0; y < MAP_SIZE; ++y)
                enhanced_traffic_field[player][x][y] = 0.0;
    for (const auto &ant : ants) {
        auto status = ant.get_status();
        if (status != Ant::Status::Alive && status != Ant::Status::Frozen)
            continue;
        enhanced_traffic_field[ant.get_player()][ant.get_x()][ant.get_y()] += 1.0;
        for (int direction = 0; direction < 6; ++direction) {
            int nx = ant.get_x() + ant_dx[ant.get_y() % 2][direction][0];
            int ny = ant.get_y() + ant_dx[ant.get_y() % 2][direction][1];
            if (ant_can_walk_to(nx, ny))
                enhanced_traffic_field[ant.get_player()][nx][ny] += 0.35;
        }
    }
}

Game::PathPlan Game::reverse_weighted_plan(
    int player, const std::vector<std::pair<int, int>> &sources,
    double damage_weight, double control_weight, double traffic_weight,
    double effect_weight) const {
    PathPlan plan;
    const double inf = std::numeric_limits<double>::infinity();
    for (int x = 0; x < MAP_SIZE; ++x)
        for (int y = 0; y < MAP_SIZE; ++y) {
            plan.total_cost[x][y] = inf;
            plan.damage_cost[x][y] = inf;
        }

    using QueueEntry = std::tuple<double, double, int, int>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> queue;
    for (const auto &[x, y] : sources) {
        if (!ant_can_walk_to(x, y))
            continue;
        if (plan.total_cost[x][y] <= 0.0)
            continue;
        plan.total_cost[x][y] = 0.0;
        plan.damage_cost[x][y] = 0.0;
        queue.push({0.0, 0.0, x, y});
    }

    while (!queue.empty()) {
        auto [current_total, current_damage, x, y] = queue.top();
        queue.pop();
        double best_total = plan.total_cost[x][y];
        double best_damage = plan.damage_cost[x][y];
        if (current_total > best_total + 1e-6)
            continue;
        if (std::abs(current_total - best_total) <= 1e-6 &&
            current_damage > best_damage + 1e-6)
            continue;

        double step_damage = cell_damage_hp(player, x, y);
        double step_control = control_risk_field[player][x][y];
        double step_traffic = enhanced_traffic_field[player][x][y];
        double step_effect = effect_pull_field[player][x][y];
        double step_total = std::max(
            MIN_PATH_STEP_COST,
            1.0 + damage_weight * step_damage +
                control_weight * step_control +
                traffic_weight * step_traffic -
                effect_weight * step_effect);

        for (int direction = 0; direction < 6; ++direction) {
            int px = x + ant_dx[y % 2][direction][0];
            int py = y + ant_dx[y % 2][direction][1];
            if (!ant_can_walk_to(px, py))
                continue;
            double next_total = current_total + step_total;
            double next_damage = current_damage + step_damage;
            if (next_total + 1e-6 < plan.total_cost[px][py] ||
                (std::abs(next_total - plan.total_cost[px][py]) <= 1e-6 &&
                 next_damage + 1e-6 < plan.damage_cost[px][py])) {
                plan.total_cost[px][py] = next_total;
                plan.damage_cost[px][py] = next_damage;
                queue.push({next_total, next_damage, px, py});
            }
        }
    }
    return plan;
}

void Game::prepare_enhanced_move_cache(bool reset_reservations) {
    refresh_static_risk_fields();
    compute_enhanced_traffic_field();
    for (int player = 0; player < 2; ++player) {
        auto worker_plan = reverse_weighted_plan(
            player,
            {{player ? PLAYER_0_BASE_CAMP_X : PLAYER_1_BASE_CAMP_X,
              player ? PLAYER_0_BASE_CAMP_Y : PLAYER_1_BASE_CAMP_Y}},
            WORKER_PATH_DAMAGE_WEIGHT,
            WORKER_PATH_CONTROL_WEIGHT,
            WORKER_PATH_TRAFFIC_WEIGHT,
            WORKER_PATH_EFFECT_WEIGHT);
        auto combat_base_plan = reverse_weighted_plan(
            player,
            {{player ? PLAYER_0_BASE_CAMP_X : PLAYER_1_BASE_CAMP_X,
              player ? PLAYER_0_BASE_CAMP_Y : PLAYER_1_BASE_CAMP_Y}},
            COMBAT_PATH_DAMAGE_WEIGHT,
            COMBAT_PATH_CONTROL_WEIGHT,
            COMBAT_PATH_TRAFFIC_WEIGHT,
            COMBAT_PATH_EFFECT_WEIGHT);
        enhanced_worker_costs[player] = worker_plan.total_cost;
        enhanced_combat_base_costs[player] = combat_base_plan.total_cost;
        enhanced_tower_plans[player].clear();
        for (const auto &tower : defensive_towers) {
            if (tower.destroy() || tower.get_player() == player)
                continue;
            std::vector<std::pair<int, int>> sources;
            for (int direction = 0; direction < 6; ++direction) {
                int nx = tower.get_x() + ant_dx[tower.get_y() % 2][direction][0];
                int ny = tower.get_y() + ant_dx[tower.get_y() % 2][direction][1];
                if (ant_can_walk_to(nx, ny))
                    sources.emplace_back(nx, ny);
            }
            if (sources.empty())
                continue;
            TowerPathPlan tower_plan;
            tower_plan.tower_id = tower.get_id();
            tower_plan.plan = reverse_weighted_plan(
                player, sources,
                COMBAT_PATH_DAMAGE_WEIGHT,
                COMBAT_PATH_CONTROL_WEIGHT,
                COMBAT_PATH_TRAFFIC_WEIGHT,
                COMBAT_PATH_EFFECT_WEIGHT);
            enhanced_tower_plans[player].push_back(tower_plan);
        }
    }

    if (reset_reservations) {
        for (int player = 0; player < 2; ++player) {
            enhanced_tower_claims[player].clear();
            for (int x = 0; x < MAP_SIZE; ++x)
                for (int y = 0; y < MAP_SIZE; ++y)
                    enhanced_reservations[player][x][y] = 0.0;
        }
    }
    enhanced_move_cells.clear();
    enhanced_move_tower_targets.clear();
    enhanced_move_cache_dirty = false;
}

void Game::ensure_enhanced_move_cache() {
    if (movement_policy != MovementPolicy::Enhanced)
        return;
    if (enhanced_move_phase_active) {
        if (enhanced_move_cache_dirty)
            prepare_enhanced_move_cache(false);
        return;
    }
    prepare_enhanced_move_cache(true);
}

double Game::tower_attack_value(const Ant &ant, const DefenseTower &tower,
                                double arrival_hp) const {
    if (arrival_hp <= 0.0)
        return -1e9;
    if (ant.is_combat_ant() && arrival_hp * 2.0 < ant.get_hp_limit()) {
        double total_damage = 0.0;
        int destroyed = 0;
        for (const auto &other : defensive_towers) {
            if (other.destroy() || other.get_player() == ant.get_player())
                continue;
            if (distance(Pos(other.get_x(), other.get_y()),
                         Pos(tower.get_x(), tower.get_y())) >
                COMBAT_SELF_DESTRUCT_RANGE)
                continue;
            total_damage += std::min(COMBAT_SELF_DESTRUCT_DAMAGE, other.get_hp());
            if (other.get_hp() <= COMBAT_SELF_DESTRUCT_DAMAGE)
                destroyed++;
        }
        return total_damage + destroyed * ATTACK_FINISH_BONUS +
               SURPLUS_HP_VALUE_WEIGHT * arrival_hp;
    }
    double direct_damage =
        static_cast<double>(std::min(ant.get_tower_attack_damage(), tower.get_hp()));
    double destroy_bonus =
        tower.get_hp() <= ant.get_tower_attack_damage() ? ATTACK_FINISH_BONUS : 0.0;
    return direct_damage + destroy_bonus + SURPLUS_HP_VALUE_WEIGHT * arrival_hp;
}

const Game::TowerPathPlan *Game::tower_plan_for(int player, int tower_id) const {
    const auto &plans = enhanced_tower_plans[player];
    for (const auto &plan : plans)
        if (plan.tower_id == tower_id)
            return &plan;
    return nullptr;
}

void Game::record_enhanced_reservation(const Ant &ant, int move) {
    if (movement_policy != MovementPolicy::Enhanced || !enhanced_move_phase_active)
        return;
    auto move_it = enhanced_move_cells.find(ant.get_id());
    if (move_it != enhanced_move_cells.end() && move != Ant::NoMove) {
        enhanced_reservations[ant.get_player()][move_it->second.first][move_it->second.second] += 1.0;
        enhanced_move_cells.erase(move_it);
    }
    auto tower_it = enhanced_move_tower_targets.find(ant.get_id());
    if (tower_it != enhanced_move_tower_targets.end()) {
        enhanced_tower_claims[ant.get_player()][tower_it->second]++;
        enhanced_move_tower_targets.erase(tower_it);
    }
}

std::vector<double> Game::directional_field_scores(
    const Ant &ant, const std::vector<std::tuple<int, int, int>> &candidates,
    const RiskField &field) const {
    std::vector<double> scores(candidates.size(),
                               field[ant.get_player()][ant.get_x()][ant.get_y()]);
    int owner[MAP_SIZE][MAP_SIZE];
    int distance_map[MAP_SIZE][MAP_SIZE];
    for (int x = 0; x < MAP_SIZE; ++x)
        for (int y = 0; y < MAP_SIZE; ++y) {
            owner[x][y] = -1;
            distance_map[x][y] = -1;
        }

    std::queue<std::pair<int, int>> queue;
    std::vector<bool> seeded(candidates.size(), false);
    double current_value = field[ant.get_player()][ant.get_x()][ant.get_y()];

    for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
        int nx = std::get<1>(candidates[index]);
        int ny = std::get<2>(candidates[index]);
        if (enemy_tower_at(ant.get_player(), nx, ny) != nullptr ||
            !ant_can_walk_to(nx, ny)) {
            scores[index] = current_value;
            continue;
        }
        if (owner[nx][ny] != -1)
            continue;
        owner[nx][ny] = index;
        distance_map[nx][ny] = 0;
        seeded[index] = true;
        queue.push({nx, ny});
    }

    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();
        int owner_index = owner[x][y];
        int next_distance = distance_map[x][y] + 1;
        for (int direction = 0; direction < 6; ++direction) {
            int nx = x + ant_dx[y % 2][direction][0];
            int ny = y + ant_dx[y % 2][direction][1];
            if (!ant_can_walk_to(nx, ny) || owner[nx][ny] != -1)
                continue;
            owner[nx][ny] = owner_index;
            distance_map[nx][ny] = next_distance;
            queue.push({nx, ny});
        }
    }

    std::vector<double> numerators(candidates.size(), 0.0);
    std::vector<double> denominators(candidates.size(), 0.0);
    double decay = ant.is_combat_ant() ? COMBAT_RISK_FIELD_DISTANCE_DECAY
                                       : WORKER_RISK_FIELD_DISTANCE_DECAY;
    for (int x = 0; x < MAP_SIZE; ++x)
        for (int y = 0; y < MAP_SIZE; ++y) {
            if (!ant_can_walk_to(x, y))
                continue;
            int owner_index = owner[x][y];
            if (owner_index < 0)
                continue;
            double weight = std::pow(decay, static_cast<double>(distance_map[x][y]));
            numerators[owner_index] += field[ant.get_player()][x][y] * weight;
            denominators[owner_index] += weight;
        }

    for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
        if (!seeded[index])
            continue;
        int nx = std::get<1>(candidates[index]);
        int ny = std::get<2>(candidates[index]);
        if (denominators[index] > 0.0)
            scores[index] = numerators[index] / denominators[index];
        else
            scores[index] = field[ant.get_player()][nx][ny];
    }
    return scores;
}

DefenseTower *Game::enemy_tower_at(int player, int x, int y) {
    if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE)
        return nullptr;
    DefenseTower *tower = map.map[x][y].tower;
    if (tower == nullptr || tower->destroy() || tower->get_player() == player)
        return nullptr;
    return tower;
}

const DefenseTower *Game::enemy_tower_at(int player, int x, int y) const {
    if (x < 0 || x >= MAP_SIZE || y < 0 || y >= MAP_SIZE)
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
        case TowerType::Bewitch:
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
        double bonus = ant.is_combat_ant() ? COMBAT_TOWER_TARGET_BONUS
                                           : WORKER_TOWER_TARGET_BONUS;
        if (ant.should_self_destruct_on_tower_attack())
            bonus += COMBAT_SELF_DESTRUCT_PULL_BONUS;
        return bonus;
    }
    if (!ant.is_combat_ant())
        return 0.0;
    double best = 0.0;
    double self_destruct_bonus =
        ant.should_self_destruct_on_tower_attack() ? COMBAT_SELF_DESTRUCT_PULL_BONUS
                                                   : 0.0;
    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.get_player() == ant.get_player())
            continue;
        double distance_score =
            std::max(0.0, COMBAT_TOWER_APPROACH_PULL_BASE -
                              distance(Pos(x, y), Pos(tower.get_x(), tower.get_y())));
        best = std::max(best, distance_score + self_destruct_bonus);
    }
    return best;
}

Pos Game::move_target_for_ant(const Ant &ant) const {
    Pos enemy = ant.get_player() ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                                 : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
    if (!ant.is_combat_ant())
        return enemy;
    const DefenseTower *best_tower = nullptr;
    int best_distance = std::numeric_limits<int>::max();
    int best_enemy_distance = std::numeric_limits<int>::max();
    int best_id = std::numeric_limits<int>::max();
    for (const auto &tower : defensive_towers) {
        if (tower.destroy() || tower.get_player() == ant.get_player())
            continue;
        int distance_to_ant =
            distance(Pos(ant.get_x(), ant.get_y()), Pos(tower.get_x(), tower.get_y()));
        int distance_to_enemy =
            distance(Pos(tower.get_x(), tower.get_y()), enemy);
        if (distance_to_ant < best_distance ||
            (distance_to_ant == best_distance &&
             (distance_to_enemy < best_enemy_distance ||
              (distance_to_enemy == best_enemy_distance &&
               tower.get_id() < best_id)))) {
            best_tower = &tower;
            best_distance = distance_to_ant;
            best_enemy_distance = distance_to_enemy;
            best_id = tower.get_id();
        }
    }
    if (best_tower != nullptr)
        return Pos(best_tower->get_x(), best_tower->get_y());
    return enemy;
}

int Game::half_plane_delta(int player, int x, int y) const {
    Pos own = player ? Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y)
                     : Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y);
    Pos enemy = player ? Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y)
                       : Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y);
    return distance(Pos(x, y), own) - distance(Pos(x, y), enemy);
}

bool Game::ant_in_own_half(const Ant &ant) const {
    return half_plane_delta(ant.get_player(), ant.get_x(), ant.get_y()) <= 0;
}

std::pair<int, int> Game::random_bewitch_target(const Ant &ant) {
    std::vector<std::pair<int, int>> cells;
    int player = ant.get_player();
    Pos own = player ? Pos(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y)
                     : Pos(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y);
    int anchor_delta = half_plane_delta(player, ant.get_x(), ant.get_y());
    for (int x = 0; x < MAP_SIZE; ++x)
        for (int y = 0; y < MAP_SIZE; ++y)
            if (ant_can_walk_to(x, y) &&
                !(x == ant.get_x() && y == ant.get_y()) &&
                half_plane_delta(player, x, y) <= anchor_delta)
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

void Game::apply_lightning_storm(Item &it, int player) {
    if (!it.duration || it.last_trigger_round == round)
        return;
    it.last_trigger_round = round;
    bool destroyed_any_tower = false;
    for (auto &ant : ants)
    {
        if (ant.get_player() == !player &&
            distance(Pos(it.x, it.y), Pos(ant.get_x(), ant.get_y())) <= 3)
        {
            ant.set_hp(-LIGHTNING_STORM_ANT_DAMAGE);
        }
    }
    if (lightning_storm_tower_strike_turn(it.duration))
    {
        for (auto &tower : defensive_towers)
        {
            if (tower.destroy() || tower.get_player() == player)
                continue;
            if (distance(Pos(it.x, it.y), Pos(tower.get_x(), tower.get_y())) > 3)
                continue;
            tower.set_changed_this_round();
            if (!tower.take_damage(LIGHTNING_STORM_TOWER_DAMAGE))
                continue;
            map.destroy(tower.get_x(), tower.get_y());
            tower.set_destroy();
            destroyed_any_tower = true;
        }
    }
    if (destroyed_any_tower)
        mark_risk_fields_dirty();
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
    case TowerType::Bewitch:
        if (!ant.is_control_immune()) {
            std::pair<int, int> target = ant_in_own_half(ant)
                                             ? std::make_pair(ant.get_player() ? PLAYER_1_BASE_CAMP_X
                                                                               : PLAYER_0_BASE_CAMP_X,
                                                              ant.get_player() ? PLAYER_1_BASE_CAMP_Y
                                                                               : PLAYER_0_BASE_CAMP_Y)
                                             : random_bewitch_target(ant);
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

int Game::choose_ant_move_legacy(const Ant &ant) {
    refresh_static_risk_fields();
    Pos target = move_target_for_ant(ant);
    std::vector<std::tuple<int, int, int>> candidates = legal_move_candidates(ant);
    if (candidates.empty())
        return -1;
    if (ant.get_behavior() == Ant::Behavior::Randomized)
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);

    std::vector<double> damage_scores =
        directional_field_scores(ant, candidates, damage_risk_field);
    std::vector<double> control_scores =
        directional_field_scores(ant, candidates, control_risk_field);
    std::vector<double> effect_scores =
        directional_field_scores(ant, candidates, effect_pull_field);
    if (ant.is_control_immune())
        std::fill(control_scores.begin(), control_scores.end(), 0.0);

    std::vector<double> scores;
    std::vector<double> raw_scores;
    std::vector<std::pair<int, int>> annotated_cells;
    std::vector<int> annotated_towers;
    scores.reserve(candidates.size());
    raw_scores.reserve(candidates.size());
    annotated_cells.reserve(candidates.size());
    annotated_towers.reserve(candidates.size());
    if (ant.get_behavior() == Ant::Behavior::Bewitched && ant.target_x >= 0 &&
        ant.target_y >= 0) {
        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            int eval_x = tower_target ? ant.get_x() : nx;
            int eval_y = tower_target ? ant.get_y() : ny;
            int index = static_cast<int>(&candidate - &candidates[0]);
            double score =
                ant.move_weights.progress *
                    move_progress_score(ant, eval_x, eval_y,
                                        Pos(ant.target_x, ant.target_y)) +
                ant.move_weights.pheromone *
                    move_pheromone_score(ant, eval_x, eval_y) -
                ant.move_weights.crowding *
                    crowding_penalty(ant, eval_x, eval_y) -
                ant.move_weights.expected_damage *
                    damage_scores[index] -
                ant.move_weights.control_risk *
                    control_scores[index] +
                ant.move_weights.tower_pull *
                    tower_pull_score(ant, eval_x, eval_y, tower_target) +
                ant.move_weights.effect_pull * effect_scores[index] +
                (tower_target ? 4.0 : 0.0);
            scores.push_back(score);
            raw_scores.push_back(score + effect_scores[index]);
        }
    } else {
        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            int eval_x = tower_target ? ant.get_x() : nx;
            int eval_y = tower_target ? ant.get_y() : ny;
            int index = static_cast<int>(&candidate - &candidates[0]);
            double progress = move_progress_score(ant, eval_x, eval_y, target);
            double pheromone = move_pheromone_score(ant, eval_x, eval_y);
            double tower_pull = tower_pull_score(ant, eval_x, eval_y, tower_target);
            double raw = progress + pheromone + tower_pull + effect_scores[index];
            raw_scores.push_back(raw);
            scores.push_back(
                ant.move_weights.progress * progress +
                ant.move_weights.pheromone * pheromone -
                ant.move_weights.crowding * crowding_penalty(ant, eval_x, eval_y) -
                ant.move_weights.expected_damage * damage_scores[index] -
                ant.move_weights.control_risk * control_scores[index] +
                ant.move_weights.tower_pull * tower_pull +
                ant.move_weights.effect_pull * effect_scores[index]);
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

int Game::choose_ant_move_enhanced(const Ant &ant) {
    ensure_enhanced_move_cache();
    std::vector<std::tuple<int, int, int>> candidates = legal_move_candidates(ant);
    if (candidates.empty())
        return Ant::NoMove;
    if (ant.get_behavior() == Ant::Behavior::Randomized)
        return std::get<0>(candidates[random_index(static_cast<int>(candidates.size()))]);
    if (ant.get_behavior() == Ant::Behavior::Bewitched)
        return choose_ant_move_legacy(ant);

    std::vector<double> scores;
    std::vector<double> raw_scores;
    std::vector<std::pair<int, int>> annotated_cells;
    std::vector<int> annotated_towers;
    scores.reserve(candidates.size());
    raw_scores.reserve(candidates.size());
    annotated_cells.reserve(candidates.size());
    annotated_towers.reserve(candidates.size());

    if (!ant.is_combat_ant()) {
        double current_cost = enhanced_worker_costs[ant.get_player()][ant.get_x()][ant.get_y()];
        double best_walk_remaining = std::numeric_limits<double>::infinity();
        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            if (enemy_tower_at(ant.get_player(), nx, ny) != nullptr)
                continue;
            best_walk_remaining =
                std::min(best_walk_remaining,
                         enhanced_worker_costs[ant.get_player()][nx][ny]);
        }
        double reroute_gain = 0.0;
        if (std::isfinite(current_cost) && std::isfinite(best_walk_remaining))
            reroute_gain = std::max(0.0, current_cost - best_walk_remaining);
        bool blocked =
            !std::isfinite(best_walk_remaining) || !std::isfinite(current_cost) ||
            (current_cost - best_walk_remaining <= WORKER_ROUTE_IMPROVEMENT_EPS);

        for (const auto &candidate : candidates) {
            int direction = std::get<0>(candidate);
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            double score = -1e9;
            if (tower_target != nullptr) {
                score = std::isfinite(current_cost) ? -current_cost : 0.0;
                score += 1.2 * std::min(ant.get_tower_attack_damage(),
                                        tower_target->get_hp());
                if (tower_target->get_hp() <= ant.get_tower_attack_damage())
                    score += ATTACK_FINISH_BONUS;
                if (blocked)
                    score += WORKER_BLOCKED_ATTACK_BONUS;
                else
                    score -= WORKER_REROUTE_ATTACK_PENALTY_WEIGHT * reroute_gain;
                auto claim_it =
                    enhanced_tower_claims[ant.get_player()].find(tower_target->get_id());
                if (claim_it != enhanced_tower_claims[ant.get_player()].end())
                    score -= WORKER_TOWER_CLAIM_WEIGHT * claim_it->second;
                score += ant.move_weights.pheromone *
                         move_pheromone_score(ant, ant.get_x(), ant.get_y());
                annotated_cells.emplace_back(-1, -1);
                annotated_towers.push_back(tower_target->get_id());
            } else {
                double remaining =
                    enhanced_worker_costs[ant.get_player()][nx][ny];
                if (std::isfinite(remaining)) {
                    score = -remaining;
                    score -= WORKER_RESERVATION_WEIGHT *
                             enhanced_reservations[ant.get_player()][nx][ny];
                    score -= 0.25 * crowding_penalty(ant, nx, ny);
                    score += ant.move_weights.pheromone *
                             move_pheromone_score(ant, nx, ny);
                }
                annotated_cells.emplace_back(nx, ny);
                annotated_towers.push_back(-1);
            }
            scores.push_back(score);
            raw_scores.push_back(score);
        }
    } else {
        std::vector<const DefenseTower *> enemy_towers;
        for (const auto &tower : defensive_towers)
            if (!tower.destroy() && tower.get_player() != ant.get_player())
                enemy_towers.push_back(&tower);

        for (const auto &candidate : candidates) {
            int nx = std::get<1>(candidate);
            int ny = std::get<2>(candidate);
            const DefenseTower *tower_target = enemy_tower_at(ant.get_player(), nx, ny);
            double score = -1e9;
            int best_tower_id = -1;
            if (tower_target != nullptr) {
                score = tower_attack_value(ant, *tower_target, ant.get_hp());
                score += ENHANCED_COMBAT_ATTACK_EXECUTION_BONUS;
                auto claim_it =
                    enhanced_tower_claims[ant.get_player()].find(tower_target->get_id());
                if (claim_it != enhanced_tower_claims[ant.get_player()].end())
                    score -= COMBAT_TOWER_CLAIM_WEIGHT * claim_it->second;
                score += ant.move_weights.pheromone *
                         move_pheromone_score(ant, ant.get_x(), ant.get_y());
                best_tower_id = tower_target->get_id();
                annotated_cells.emplace_back(-1, -1);
            } else if (!enemy_towers.empty()) {
                for (const DefenseTower *tower : enemy_towers) {
                    const TowerPathPlan *plan =
                        tower_plan_for(ant.get_player(), tower->get_id());
                    if (plan == nullptr)
                        continue;
                    double travel_cost = plan->plan.total_cost[nx][ny];
                    if (!std::isfinite(travel_cost))
                        continue;
                    double travel_damage = plan->plan.damage_cost[nx][ny];
                    double arrival_hp = ant.get_hp() - travel_damage;
                    double utility = tower_attack_value(ant, *tower, arrival_hp);
                    utility -= COMBAT_TRAVEL_COST_WEIGHT * travel_cost;
                    auto claim_it =
                        enhanced_tower_claims[ant.get_player()].find(tower->get_id());
                    if (claim_it != enhanced_tower_claims[ant.get_player()].end())
                        utility -= COMBAT_TOWER_CLAIM_WEIGHT * claim_it->second;
                    if (utility > score) {
                        score = utility;
                        best_tower_id = tower->get_id();
                    }
                }
                if (std::isfinite(score)) {
                    score -= COMBAT_RESERVATION_WEIGHT *
                             enhanced_reservations[ant.get_player()][nx][ny];
                    score += ant.move_weights.pheromone *
                             move_pheromone_score(ant, nx, ny);
                }
                annotated_cells.emplace_back(nx, ny);
            } else {
                double remaining =
                    enhanced_combat_base_costs[ant.get_player()][nx][ny];
                if (std::isfinite(remaining)) {
                    score = -remaining;
                    score -= COMBAT_RESERVATION_WEIGHT *
                             enhanced_reservations[ant.get_player()][nx][ny];
                    score += ant.move_weights.pheromone *
                             move_pheromone_score(ant, nx, ny);
                }
                annotated_cells.emplace_back(nx, ny);
            }
            annotated_towers.push_back(best_tower_id);
            scores.push_back(score);
            raw_scores.push_back(score);
        }
    }

    auto commit_annotation = [&](int index) {
        enhanced_move_cells.erase(ant.get_id());
        enhanced_move_tower_targets.erase(ant.get_id());
        if (index < 0 || index >= static_cast<int>(annotated_cells.size()))
            return;
        if (annotated_cells[index].first >= 0)
            enhanced_move_cells[ant.get_id()] = annotated_cells[index];
        if (annotated_towers[index] >= 0)
            enhanced_move_tower_targets[ant.get_id()] = annotated_towers[index];
    };

    if (ant.get_behavior() == Ant::Behavior::Conservative ||
        ant.get_behavior() == Ant::Behavior::ControlFree) {
        int best = 0;
        for (int i = 1; i < static_cast<int>(scores.size()); ++i)
            if (scores[i] > scores[best] ||
                (scores[i] == scores[best] && raw_scores[i] > raw_scores[best]))
                best = i;
        commit_annotation(best);
        return std::get<0>(candidates[best]);
    }
    double max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> probs(scores.size(), 0.0);
    double total = 0.0;
    for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
        probs[i] = std::exp((scores[i] - max_score) / DEFAULT_MOVE_TEMPERATURE);
        total += probs[i];
    }
    if (total <= 0.0)
    {
        commit_annotation(0);
        return std::get<0>(candidates[0]);
    }
    double threshold = random_float();
    double cumulative = 0.0;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        cumulative += probs[i] / total;
        if (threshold <= cumulative) {
            commit_annotation(i);
            return std::get<0>(candidates[i]);
        }
    }
    commit_annotation(static_cast<int>(candidates.size()) - 1);
    return std::get<0>(candidates.back());
}

int Game::choose_ant_move(const Ant &ant) {
    if (movement_policy == MovementPolicy::Legacy)
        return choose_ant_move_legacy(ant);
    return choose_ant_move_enhanced(ant);
}

void Game::attack_tower_from_ant(Ant &ant, DefenseTower &tower) {
    if (ant.should_self_destruct_on_tower_attack()) {
        const Pos blast_center(tower.get_x(), tower.get_y());
        bool destroyed_any_tower = false;
        for (auto &other : defensive_towers) {
            if (other.destroy() || other.get_player() == ant.get_player())
                continue;
            if (distance(blast_center, Pos(other.get_x(), other.get_y())) >
                COMBAT_SELF_DESTRUCT_RANGE)
                continue;
            other.set_changed_this_round();
            if (!other.take_damage(COMBAT_SELF_DESTRUCT_DAMAGE))
                continue;
            map.destroy(other.get_x(), other.get_y());
            other.set_destroy();
            destroyed_any_tower = true;
        }
        if (destroyed_any_tower)
            mark_risk_fields_dirty();
        ant.set_hp_true(-ant.get_hp());
        return;
    }
    if (tower.take_damage(ant.get_tower_attack_damage())) {
        tower.set_changed_this_round();
        map.destroy(tower.get_x(), tower.get_y());
        tower.set_destroy();
        mark_risk_fields_dirty();
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

void Game::resolve_random_move_steps(Ant &ant, int steps) {
    for (int step = 0; step < steps; ++step) {
        auto status = ant.get_status();
        if (status == Ant::Status::Fail || status == Ant::Status::TooOld)
            break;
        int move = random_index(3) < 2 ? choose_random_legal_move(ant)
                                       : choose_ant_move(ant);
        resolve_ant_step(ant, move);
        invalidate_enhanced_move_cache();
    }
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
    std::vector<Ant *> chosen;
    while (!eligible.empty() && teleport_count-- > 0) {
        int ant_idx = random_index(static_cast<int>(eligible.size()));
        chosen.push_back(eligible[ant_idx]);
        eligible.erase(eligible.begin() + ant_idx);
    }
    for (Ant *ant : chosen)
        resolve_random_move_steps(*ant, 3);
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
    movement_policy = parse_movement_policy(config);
    cold_handle_rule_illegal = parse_cold_handle_rule_illegal(config);
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
    mark_risk_fields_dirty();
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
        apply_lightning_storm(it, player);
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
                int splash = tower.get_range();
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
                                 Pos(tower.get_x(), tower.get_y())) <=
                        tower.get_range()) {
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
    begin_move_phase();
    for (auto &ant : ants)
    {
        int move = -1;

        if (ant.get_status() == Ant::Status::Alive) {
            move = choose_ant_move(ant);
            record_enhanced_reservation(ant, move);
        }
        resolve_ant_step(ant, move);
    }
    end_move_phase();
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
    if ((round + 1) % 2 != 0)
        return;
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

int Game::tower_count_for_player(int player) const {
    int count = 0;
    for (const auto &tower : defensive_towers) {
        if (!tower.destroy() && tower.get_player() == player)
            count++;
    }
    return count;
}

bool Game::apply_operation(const std::vector<Operation> &op_list, int player,
                           std::string &err_msg,
                           Game::OperationErrorKind *error_kind)
{
    if (error_kind != nullptr)
        *error_kind = OperationErrorKind::None;
    auto fail = [&](OperationErrorKind kind, const std::string &message) {
        err_msg = message;
        if (error_kind != nullptr)
            *error_kind = kind;
        return false;
    };
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
                return fail(OperationErrorKind::Rule, msg);
            }
            if (map.map[x][y].base_camp != nullptr)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: attempt to build a tower (at %d, %d), in which there is already a camp. (player id = %d)",
                        x, y, map.map[x][y].player);
                return fail(OperationErrorKind::Rule, msg);
            }
            if (map.map[x][y].tower != nullptr)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: attempt to build a tower (at %d, %d), in which there is already a tower. (player id = %d)",
                        x, y, map.map[x][y].player);
                return fail(OperationErrorKind::Rule, msg);
            }
            if (map.map[x][y].player != player)
            {
                char msg[100];
                ;
                sprintf(msg, "TowerBuild: Build a tower at position (%d, %d), its player is %d, request player = %d", x, y, map.map[x][y].player, player);
                return fail(OperationErrorKind::Rule, msg);
            }
            const int active_tower_count = tower_count_for_player(player);
            if (player == 1 &&
                !player1.coin.isEnough_tower_build(active_tower_count))
            { // not enough money
                return fail(OperationErrorKind::Rule,
                            "TowerBuild: P1 not enough money");
            }
            if (player == 0 &&
                !player0.coin.isEnough_tower_build(active_tower_count))
            {
                return fail(OperationErrorKind::Rule,
                            "TowerBuild: P0 not enough money");
            }
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration && distance(Pos(x, y), Pos(it.x, it.y)) <= 3)
            {
                return fail(OperationErrorKind::Rule,
                            "TowerBuild: EMPBlaster is active");
            }

            if (player == 1)
                player1.coin.cost_tower_build(active_tower_count);
            else
                player0.coin.cost_tower_build(active_tower_count);

            used_tower.push_back(tower_id);
            defensive_towers.push_back(DefenseTower{x, y, player, tower_id, 0});
            DefenseTower &new_tower = defensive_towers.back();
            map.build(&new_tower);
            new_tower.set_changed_this_round();
            mark_risk_fields_dirty();
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
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: Invalid Tower id");
            }
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
            {
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: Tower has been used");
            }
            DefenseTower &tower = defensive_towers[id];
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration &&
                distance(Pos(tower.get_x(), tower.get_y()), Pos(it.x, it.y)) <= 3)
            {
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: EMPBlaster is active");
            }

            if (tower.get_level() ==
                TOWER_MAX_LEVEL)
            { // have reached max level
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: Tower has reached max level");
            }
            if (player == 1 && !player1.coin.isEnough_tower_upgrade(
                                   tower))
            { // not enough money
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: P1 not enough money");
            }
            if (player == 0 && !player0.coin.isEnough_tower_upgrade(tower))
            {
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: P0 not enough money");
            }
            if (!tower.upgrade_type_check(op.get_args()))
            {
                return fail(OperationErrorKind::Rule,
                            "TowerUpgrade: Invalid upgrade type");
            }

            used_tower.push_back(id);
            if (player == 1) // must cost coin first!!
                player1.coin.cost_tower_upgrade(tower);
            else
                player0.coin.cost_tower_upgrade(tower);

            tower.upgrade(TowerType(op.get_args()));
            tower.set_changed_this_round();
            mark_risk_fields_dirty();
            // output.add_tower(tower, op.get_args());
            break;
        }
        case Operation::Type::TowerDestroy:
        {
            int id = op.get_id();
            if (id < 0 || id >= (int)defensive_towers.size() ||
                defensive_towers[id].destroy() || defensive_towers[id].get_player() != player)
            {
                return fail(OperationErrorKind::Rule,
                            "TowerDestroy: Invalid Tower id");
            }
            if (std::find(used_tower.begin(), used_tower.end(), id) !=
                used_tower.end())
            {
                return fail(OperationErrorKind::Rule,
                            "TowerDestroy: Tower has been used");
            }
            DefenseTower *defensive_tower = &defensive_towers[id];
            Item it = item[!player][ItemType::EMPBlaster];
            if (it.duration &&
                distance(Pos(defensive_tower->get_x(), defensive_tower->get_y()),
                         Pos(it.x, it.y)) <= 3)
            {
                return fail(OperationErrorKind::Rule,
                            "TowerDestroy: EMPBlaster is active");
            }
            const int active_tower_count = tower_count_for_player(player);
            if (player == 1)
                player1.coin.income_tower_destroy(*defensive_tower,
                                                  active_tower_count);
            else
                player0.coin.income_tower_destroy(*defensive_tower,
                                                  active_tower_count);

            if (defensive_tower->get_type() == TowerType::Basic)
            {
                map.destroy(defensive_tower->get_x(), defensive_tower->get_y());
                output.add_tower(*defensive_tower, TOWER_DESTROY_TYPE,
                                 defensive_tower->get_attack());
                defensive_tower->set_destroy();
                mark_risk_fields_dirty();
            }
            else
            {
                TowerType new_type = defensive_tower->tower_downgrade_type();
                defensive_tower->downgrade(new_type);
                defensive_tower->set_changed_this_round();
                mark_risk_fields_dirty();
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
            apply_lightning_storm(item[player][it], player);
            mark_risk_fields_dirty();

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
            mark_risk_fields_dirty();
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
            mark_risk_fields_dirty();

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
            mark_risk_fields_dirty();

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
            return fail(OperationErrorKind::Protocol,
                        "Operation: undefined operation type");
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
    output.add_weapon_cooldowns(item[0], item[1]);
    output.add_active_effects(item[0], item[1]);
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
    output.add_weapon_cooldowns(item[0], item[1]);
    output.add_active_effects(item[0], item[1]);
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
        if (player == 0)
        {
            err_msg.clear();
            op[0].clear();
            op[1].clear();
        }
        judger_round_info.transfer_op(output_to_judger.if_ai(player));

        if (judger_round_info.get_player() != player)
        {
            is_end = true;
            winner = player;
            return false;
        }

        int another_player = 1 - player;

        std::vector<Operation> op_list = judger_round_info.get_op_list();
        if (!cold_handle_rule_illegal)
        {
            op[player] = op_list;
            if (!apply_operation(op_list, player, err_msg))
            {
                set_AI_state_IO(player);
                return false;
            }
        }
        else
        {
            std::vector<Operation> accepted_ops;
            std::vector<int> used_tower;
            std::vector<std::string> ignored_errors;
            bool camp_upgraded_flag = false;
            for (const auto &operation : op_list)
            {
                const auto type = operation.get_operation_type();
                if ((type == Operation::Type::TowerUpgrade ||
                     type == Operation::Type::TowerDestroy) &&
                    std::find(used_tower.begin(), used_tower.end(),
                              operation.get_id()) != used_tower.end())
                {
                    ignored_errors.push_back(
                        type == Operation::Type::TowerUpgrade
                            ? "TowerUpgrade: Tower has been used"
                            : "TowerDestroy: Tower has been used");
                    continue;
                }
                if (is_base_upgrade_operation(type) && camp_upgraded_flag)
                {
                    ignored_errors.push_back(
                        "BarrackUpgrade: already upgraded this tern");
                    continue;
                }
                const int pending_tower_id = tower_id;
                std::string operation_error;
                OperationErrorKind error_kind = OperationErrorKind::None;
                if (!apply_operation(std::vector<Operation>{operation}, player,
                                     operation_error, &error_kind))
                {
                    if (error_kind == OperationErrorKind::Protocol)
                    {
                        op[player] = op_list;
                        err_msg = operation_error;
                        set_AI_state_IO(player);
                        return false;
                    }
                    ignored_errors.push_back(operation_error);
                    continue;
                }
                accepted_ops.push_back(operation);
                if (type == Operation::Type::TowerBuild)
                    used_tower.push_back(pending_tower_id);
                else if (type == Operation::Type::TowerUpgrade ||
                         type == Operation::Type::TowerDestroy)
                    used_tower.push_back(operation.get_id());
                if (is_base_upgrade_operation(type))
                    camp_upgraded_flag = true;
            }
            op[player] = accepted_ops;
            judger_round_info.set_operation_list(accepted_ops);
            append_illegal_summary(err_msg, player, ignored_errors);
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
