#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define private public
#include "../game/include/game.hpp"
#undef private

namespace py = pybind11;

namespace {

constexpr int INITIAL_COIN = 50;
constexpr int SPECIAL_BEHAVIOR_DECAY_TURNS = 5;

int tower_build_cost_for_count(int tower_count) {
    tower_count = std::max(tower_count, 0);
    int cost = 15;
    for (int index = 0; index < tower_count / 2; ++index)
        cost *= 3;
    if (tower_count % 2 == 1)
        cost *= 2;
    return cost;
}

int default_behavior_expiry(Ant::Behavior behavior) {
    switch (behavior) {
    case Ant::Behavior::Conservative:
    case Ant::Behavior::Bewitched:
    case Ant::Behavior::ControlFree:
        return SPECIAL_BEHAVIOR_DECAY_TURNS;
    default:
        return 0;
    }
}

struct BoundOperation {
    int type;
    int arg0;
    int arg1;

    BoundOperation(int type_ = -1, int arg0_ = -1, int arg1_ = -1) : type(type_), arg0(arg0_), arg1(arg1_) {}
};

::Operation to_game_operation(const BoundOperation &operation) {
    switch (operation.type) {
    case 11:
    case 21:
    case 22:
    case 23:
    case 24:
        return ::Operation(operation.type, -1, -1, operation.arg0, operation.arg1);
    case 12:
        return ::Operation(operation.type, operation.arg0, operation.arg1, -1, -1);
    case 13:
        return ::Operation(operation.type, operation.arg0, -1, -1, -1);
    case 31:
    case 32:
        return ::Operation(operation.type, -1, -1, -1, -1);
    default:
        return ::Operation();
    }
}

void reset_items(Game &game) {
    for (int player = 0; player < 2; ++player) {
        game.item[player].clear();
        for (int index = 0; index < ItemType::Count; ++index)
            game.item[player].emplace_back(0, 0, 0, 0);
    }
}

void rewire_map(Game &game) {
    for (int x = 0; x < MAP_SIZE; ++x) {
        for (int y = 0; y < MAP_SIZE; ++y) {
            game.map.map[x][y].tower = nullptr;
            game.map.map[x][y].base_camp = nullptr;
        }
    }
    game.map.map[PLAYER_0_BASE_CAMP_X][PLAYER_0_BASE_CAMP_Y].base_camp = &game.base_camp0;
    game.map.map[PLAYER_1_BASE_CAMP_X][PLAYER_1_BASE_CAMP_Y].base_camp = &game.base_camp1;
    for (auto &tower : game.defensive_towers) {
        if (tower.destroy())
            continue;
        game.map.map[tower.get_x()][tower.get_y()].tower = &tower;
    }
}

Game::MovementPolicy parse_movement_policy_name(const std::string &policy_name) {
    if (policy_name == "legacy")
        return Game::MovementPolicy::Legacy;
    return Game::MovementPolicy::Enhanced;
}

std::string movement_policy_name(Game::MovementPolicy policy) {
    return policy == Game::MovementPolicy::Legacy ? "legacy" : "enhanced";
}

void init_game(Game &game, unsigned long long seed,
               Game::MovementPolicy movement_policy,
               bool cold_handle_rule_illegal) {
    game.is_end = false;
    game.winner = -1;
    game.round = 0;
    game.ant_id = 0;
    game.barrack_id = 0;
    game.tower_id = 0;
    game.err_msg.clear();
    game.random_seed = seed;
    game.movement_policy = movement_policy;
    game.cold_handle_rule_illegal = cold_handle_rule_illegal;
    game.enhanced_move_phase_active = false;
    game.enhanced_move_cache_dirty = true;
    game.rng_state = seed & ((1ULL << 48) - 1);
    game.record_file.clear();
    game.player0 = Player();
    game.player1 = Player();
    game.player0.ant_target_x = PLAYER_1_BASE_CAMP_X;
    game.player0.ant_target_y = PLAYER_1_BASE_CAMP_Y;
    game.player1.ant_target_x = PLAYER_0_BASE_CAMP_X;
    game.player1.ant_target_y = PLAYER_0_BASE_CAMP_Y;
    game.player0.coin.coin = INITIAL_COIN;
    game.player1.coin.coin = INITIAL_COIN;
    game.player0.coin.basic_income = 3;
    game.player1.coin.basic_income = 3;
    game.player0.coin.tower_building_price = tower_build_cost_for_count(0);
    game.player1.coin.tower_building_price = tower_build_cost_for_count(0);
    game.player0.coin.penalty = 0;
    game.player1.coin.penalty = 0;
    game.map = Map();
    game.map.init_pheromon(seed);
    game.base_camp0 = Headquarter(PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y, 0, 0, 0, 50);
    game.base_camp1 = Headquarter(PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y, 1, 0, 0, 50);
    game.defensive_towers.clear();
    game.ants.clear();
    game.op[0].clear();
    game.op[1].clear();
    reset_items(game);
    game.state[0] = Game::AI_state::OK;
    game.state[1] = Game::AI_state::OK;
    rewire_map(game);
}

int tower_level_from_type(TowerType tower_type) {
    if (tower_type == TowerType::Basic)
        return 0;
    return static_cast<int>(tower_type) < 10 ? 1 : 2;
}

int display_cooldown_to_round(const DefenseTower &tower, int cooldown) {
    const int speed = static_cast<int>(std::llround(tower.get_spd()));
    if (tower.get_spd() < 1.0)
        return 0;
    return std::max(0, speed - cooldown);
}

std::vector<std::vector<int>> tower_rows(const Game &game) {
    std::vector<std::vector<int>> rows;
    rows.reserve(game.defensive_towers.size());
    for (const auto &tower : game.defensive_towers) {
        if (tower.destroy())
            continue;
        rows.push_back({
            tower.get_id(),
            tower.get_player(),
            tower.get_x(),
            tower.get_y(),
            static_cast<int>(tower.get_type()),
            tower.get_cd(),
            tower.get_hp(),
        });
    }
    return rows;
}

std::vector<std::vector<int>> ant_rows(const Game &game) {
    std::vector<std::vector<int>> rows;
    rows.reserve(game.ants.size());
    for (const auto &ant : game.ants) {
        rows.push_back({
            ant.get_id(),
            ant.get_player(),
            ant.get_x(),
            ant.get_y(),
            ant.get_hp(),
            ant.get_level(),
            ant.age,
            static_cast<int>(ant.get_status()),
            static_cast<int>(ant.get_behavior()),
            static_cast<int>(ant.get_kind()),
        });
    }
    return rows;
}

std::vector<std::vector<int>> base_rows(const Game &game) {
    return {
        {0, game.base_camp0.get_x(), game.base_camp0.get_y(), game.base_camp0.get_hp(), game.base_camp0.get_cd_level(), game.base_camp0.get_ant_level()},
        {1, game.base_camp1.get_x(), game.base_camp1.get_y(), game.base_camp1.get_hp(), game.base_camp1.get_cd_level(), game.base_camp1.get_ant_level()},
    };
}

std::vector<int> coin_rows(const Game &game) { return {game.player0.coin.get_coin(), game.player1.coin.get_coin()}; }

std::vector<int> die_count_rows(const Game &game) {
    return {game.player1.opponent_killed_ant, game.player0.opponent_killed_ant};
}

std::vector<int> super_weapon_usage_rows(const Game &game) {
    return {game.player0.super_weapons_usage, game.player1.super_weapons_usage};
}

std::vector<int> ai_time_rows(const Game &game) {
    return {game.player0.AI_total_time, game.player1.AI_total_time};
}

std::vector<std::vector<int>> weapon_cooldown_rows(const Game &game) {
    std::vector<std::vector<int>> rows(2, std::vector<int>(5, 0));
    for (int player = 0; player < 2; ++player) {
        for (int item = 0; item < ItemType::Count; ++item)
            rows[player][item + 1] = game.item[player][item].cd;
    }
    return rows;
}

std::vector<std::vector<int>> effect_rows(const Game &game) {
    std::vector<std::vector<int>> rows;
    for (int player = 0; player < 2; ++player) {
        for (int item = 0; item < ItemType::Count; ++item) {
            const auto &effect = game.item[player][item];
            if (effect.duration <= 0)
                continue;
            rows.push_back({item + 1, player, effect.x, effect.y, effect.duration});
        }
    }
    return rows;
}

bool is_tower_operation(int type) { return type == 11 || type == 12 || type == 13; }

bool is_base_upgrade_operation(int type) { return type == 31 || type == 32; }

void sync_terminal(Game &game, bool &terminal, int &winner) {
    terminal = game.is_end;
    winner = terminal ? game.winner : -1;
}

} // namespace

struct NativeState {
    Game game;
    bool terminal = false;
    int winner = -1;
    unsigned long long seed = 0;
    bool cold_handle_rule_illegal = false;
    std::array<int, 2> old_count = {0, 0};

    explicit NativeState(unsigned long long init_seed,
                         const std::string &movement_policy_name_in = "enhanced",
                         bool cold_handle_rule_illegal_in = false)
        : seed(init_seed),
          cold_handle_rule_illegal(cold_handle_rule_illegal_in) {
        init_game(game, seed, parse_movement_policy_name(movement_policy_name_in),
                  cold_handle_rule_illegal);
    }

    NativeState clone() const {
        NativeState copy(*this);
        rewire_map(copy.game);
        return copy;
    }

    int round_index() const { return game.round; }

    std::vector<int> coins() const { return coin_rows(game); }

    std::vector<int> old_count_rows() const { return {old_count[0], old_count[1]}; }

    std::string movement_policy_name_view() const {
        return movement_policy_name(game.movement_policy);
    }

    std::vector<int> die_count() const { return die_count_rows(game); }

    std::vector<int> super_weapon_usage() const { return super_weapon_usage_rows(game); }

    std::vector<int> ai_time() const { return ai_time_rows(game); }

    std::vector<std::vector<int>> weapon_cooldowns() const { return weapon_cooldown_rows(game); }

    std::vector<std::vector<int>> tower_rows_view() const { return tower_rows(game); }

    std::vector<std::vector<int>> ant_rows_view() const { return ant_rows(game); }

    std::vector<std::vector<int>> base_rows_view() const { return base_rows(game); }

    std::vector<std::vector<int>> effect_rows_view() const { return effect_rows(game); }

    int next_ant_id() const { return game.ant_id; }

    int next_tower_id() const { return game.tower_id; }

    std::vector<BoundOperation> apply_operation_list(int player_id, const std::vector<BoundOperation> &operations) {
        std::vector<BoundOperation> illegal;
        illegal.reserve(operations.size());
        std::unordered_set<int> used_towers;
        bool base_upgraded = false;
        for (const auto &operation : operations) {
            if ((operation.type == 12 || operation.type == 13) && used_towers.find(operation.arg0) != used_towers.end()) {
                illegal.push_back(operation);
                if (!cold_handle_rule_illegal) {
                    game.is_end = true;
                    game.winner = 1 - player_id;
                    break;
                }
                continue;
            }
            if (is_base_upgrade_operation(operation.type) && base_upgraded) {
                illegal.push_back(operation);
                if (!cold_handle_rule_illegal) {
                    game.is_end = true;
                    game.winner = 1 - player_id;
                    break;
                }
                continue;
            }
            const int pending_tower_id = game.tower_id;
            std::string err_msg;
            Game::OperationErrorKind error_kind = Game::OperationErrorKind::None;
            if (!game.apply_operation(std::vector<::Operation>{to_game_operation(operation)}, player_id, err_msg, &error_kind)) {
                illegal.push_back(operation);
                if (error_kind == Game::OperationErrorKind::Protocol ||
                    !cold_handle_rule_illegal) {
                    game.is_end = true;
                    game.winner = 1 - player_id;
                    break;
                }
                continue;
            }
            if (operation.type == 11)
                used_towers.insert(pending_tower_id);
            else if (operation.type == 12 || operation.type == 13)
                used_towers.insert(operation.arg0);
            if (is_base_upgrade_operation(operation.type))
                base_upgraded = true;
        }
        sync_terminal(game, terminal, winner);
        return illegal;
    }

    py::dict advance_round() {
        if (game.is_end) {
            sync_terminal(game, terminal, winner);
            py::dict out;
            out["terminal"] = terminal;
            out["winner"] = winner;
            return out;
        }
        game.next_round();
        sync_terminal(game, terminal, winner);
        py::dict out;
        out["terminal"] = terminal;
        out["winner"] = winner;
        return out;
    }

    py::dict resolve_turn(const std::vector<BoundOperation> &ops0, const std::vector<BoundOperation> &ops1) {
        auto illegal0 = apply_operation_list(0, ops0);
        std::vector<BoundOperation> illegal1;
        if (!game.is_end)
            illegal1 = apply_operation_list(1, ops1);
        py::dict out;
        if (!game.is_end)
            out = advance_round();
        else
            sync_terminal(game, terminal, winner);
        out["terminal"] = terminal;
        out["winner"] = winner;
        out["illegal0"] = illegal0;
        out["illegal1"] = illegal1;
        return out;
    }

    void sync_public_round_state(
        int round,
        const std::vector<std::vector<int>> &tower_rows_in,
        const std::vector<std::vector<int>> &ant_rows_in,
        const std::vector<int> &coins_in,
        const std::vector<int> &camps_hp,
        const std::vector<int> &speed_lv,
        const std::vector<int> &anthp_lv,
        const std::vector<std::vector<int>> &weapon_cooldowns_in,
        const std::vector<std::vector<int>> &active_effect_rows_in) {
        const std::unordered_map<int, Ant> previous_ants = [&]() {
            std::unordered_map<int, Ant> ants_by_id;
            for (const auto &ant : game.ants)
                ants_by_id.emplace(ant.get_id(), ant);
            return ants_by_id;
        }();

        game.round = round;
        if (coins_in.size() >= 2) {
            game.player0.coin.coin = coins_in[0];
            game.player1.coin.coin = coins_in[1];
        }
        if (camps_hp.size() >= 2) {
            game.base_camp0.hp = camps_hp[0];
            game.base_camp1.hp = camps_hp[1];
        }
        if (speed_lv.size() >= 2) {
            game.base_camp0.cd_level = speed_lv[0];
            game.base_camp1.cd_level = speed_lv[1];
        }
        if (anthp_lv.size() >= 2) {
            game.base_camp0.ant_level = anthp_lv[0];
            game.base_camp1.ant_level = anthp_lv[1];
        }

        game.defensive_towers.clear();
        std::array<int, 2> tower_counts = {0, 0};
        int max_tower_id = 0;
        for (const auto &row : tower_rows_in) {
            if (row.size() < 6)
                continue;
            const int tower_id = row[0];
            const int player = row[1];
            const int x = row[2];
            const int y = row[3];
            const TowerType tower_type = static_cast<TowerType>(row[4]);
            const int cooldown = row[5];
            const int hp = row.size() >= 7 ? row[6] : 10;
            game.defensive_towers.emplace_back(x, y, player, tower_id, 0);
            auto &tower = game.defensive_towers.back();
            if (tower_type != TowerType::Basic)
                tower.upgrade(tower_type);
            tower.level = tower_level_from_type(tower_type);
            tower.round = display_cooldown_to_round(tower, cooldown);
            tower.hp = hp;
            tower.changed = false;
            tower.attacked_ants.clear();
            if (player >= 0 && player < 2)
                tower_counts[player] += 1;
            max_tower_id = std::max(max_tower_id, tower_id + 1);
        }
        game.tower_id = max_tower_id;
        game.player0.coin.tower_building_price = tower_build_cost_for_count(tower_counts[0]);
        game.player1.coin.tower_building_price = tower_build_cost_for_count(tower_counts[1]);

        game.ants.clear();
        int max_ant_id = 0;
        for (const auto &row : ant_rows_in) {
            if (row.size() < 8)
                continue;
            const int ant_id = row[0];
            const int player = row[1];
            const int x = row[2];
            const int y = row[3];
            const int hp = row[4];
            const int level = row[5];
            const int public_age = row[6];
            const Ant::Kind kind =
                row.size() >= 10 ? static_cast<Ant::Kind>(row[9])
                                 : Ant::Kind::Worker;
            game.ants.emplace_back(player, ant_id, x, y, level, kind);
            auto &ant = game.ants.back();
            auto it = previous_ants.find(ant_id);
            if (it != previous_ants.end()) {
                ant.trail_cells = it->second.trail_cells;
                ant.last_move = it->second.last_move;
                ant.path_len_total = it->second.path_len_total;
                ant.age = public_age;
                ant.shield = it->second.shield;
                ant.defend = it->second.defend;
                ant.evasion = it->second.evasion;
            } else {
                ant.age = public_age;
                ant.shield = 0;
                ant.defend = false;
                ant.evasion = false;
            }
            ant.pos_x = x;
            ant.pos_y = y;
            ant.hp = hp;
            ant.is_frozen = (row[7] == static_cast<int>(Ant::Status::Frozen));
            ant.all_frozen = ant.is_frozen;
            if (it != previous_ants.end()) {
                ant.behavior = it->second.behavior;
                ant.behavior_rounds = it->second.behavior_rounds;
                ant.behavior_expiry = it->second.behavior_expiry;
                ant.target_x = it->second.target_x;
                ant.target_y = it->second.target_y;
                ant.has_pending_behavior = it->second.has_pending_behavior;
                ant.pending_behavior = it->second.pending_behavior;
            }
            if (row.size() >= 9) {
                const Ant::Behavior public_behavior =
                    static_cast<Ant::Behavior>(row[8]);
                if (ant.behavior != public_behavior) {
                    ant.behavior_rounds = 0;
                    ant.behavior_expiry =
                        default_behavior_expiry(public_behavior);
                }
                ant.behavior = public_behavior;
                if (ant.behavior != Ant::Behavior::Bewitched) {
                    ant.target_x = -1;
                    ant.target_y = -1;
                }
            }
            ant.set_kind(kind);
            max_ant_id = std::max(max_ant_id, ant_id + 1);
        }
        game.ant_id = max_ant_id;

        reset_items(game);
        for (int player = 0; player < std::min<int>(2, weapon_cooldowns_in.size());
             ++player) {
            const auto &row = weapon_cooldowns_in[player];
            for (int index = 0; index < std::min<int>(ItemType::Count, row.size());
                 ++index) {
                game.item[player][index].cd = row[index];
            }
        }
        for (const auto &row : active_effect_rows_in) {
            if (row.size() < 5)
                continue;
            const int item_type = row[0] - 1;
            const int player = row[1];
            if (player < 0 || player >= 2 || item_type < 0 ||
                item_type >= ItemType::Count) {
                continue;
            }
            Item &item = game.item[player][item_type];
            item.x = row[2];
            item.y = row[3];
            item.duration = row[4];
        }

        game.is_end = false;
        game.winner = -1;
        if (game.base_camp0.get_hp() <= 0 || game.base_camp1.get_hp() <= 0) {
            game.judge_base_camp();
        } else if (game.round >= MAX_ROUND) {
            game.is_end = true;
            game.judge_winner();
        }
        rewire_map(game);
        sync_terminal(game, terminal, winner);
    }
};

PYBIND11_MODULE(native_antwar, m) {
    py::class_<BoundOperation>(m, "Operation")
        .def(py::init<int, int, int>(), py::arg("type"), py::arg("arg0") = -1, py::arg("arg1") = -1)
        .def_readwrite("type", &BoundOperation::type)
        .def_readwrite("arg0", &BoundOperation::arg0)
        .def_readwrite("arg1", &BoundOperation::arg1);

    py::class_<NativeState>(m, "NativeState")
        .def(py::init<unsigned long long, const std::string &, bool>(),
             py::arg("seed"),
             py::arg("movement_policy") = "enhanced",
             py::arg("cold_handle_rule_illegal") = false)
        .def("clone", &NativeState::clone)
        .def_readwrite("terminal", &NativeState::terminal)
        .def_readwrite("winner", &NativeState::winner)
        .def_readonly("seed", &NativeState::seed)
        .def_readonly("cold_handle_rule_illegal", &NativeState::cold_handle_rule_illegal)
        .def_property_readonly("movement_policy", &NativeState::movement_policy_name_view)
        .def("round_index", &NativeState::round_index)
        .def("coins", &NativeState::coins)
        .def("old_count", &NativeState::old_count_rows)
        .def("die_count", &NativeState::die_count)
        .def("super_weapon_usage", &NativeState::super_weapon_usage)
        .def("ai_time", &NativeState::ai_time)
        .def("weapon_cooldowns", &NativeState::weapon_cooldowns)
        .def("tower_rows", &NativeState::tower_rows_view)
        .def("ant_rows", &NativeState::ant_rows_view)
        .def("base_rows", &NativeState::base_rows_view)
        .def("effect_rows", &NativeState::effect_rows_view)
        .def("next_ant_id", &NativeState::next_ant_id)
        .def("next_tower_id", &NativeState::next_tower_id)
        .def("apply_operation_list", &NativeState::apply_operation_list)
        .def("advance_round", &NativeState::advance_round)
        .def("resolve_turn", &NativeState::resolve_turn)
        .def("sync_public_round_state", &NativeState::sync_public_round_state);
}
