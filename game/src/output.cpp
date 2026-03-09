#include "../include/output.h"
#include <fstream>
#include <iomanip>
#include <type_traits>

using json = nlohmann::json;

// Return an empty record for cur.
json Output::new_record() {
    return json{
        {"towers", json::array()},
        {"ants", json::array()},
        {"towers", json::array()},
    };
}

Output::Output() : data(), cur(new_record()) {}

// Serialize Ant::Status.
NLOHMANN_JSON_SERIALIZE_ENUM(Ant::Status, {{Ant::Status::Alive, 0},
                                           {Ant::Status::Success, 1},
                                           {Ant::Status::Fail, 2},
                                           {Ant::Status::TooOld, 3},
                                           {Ant::Status::Frozen, 4}})

// Serialize Ant::Behavior.
NLOHMANN_JSON_SERIALIZE_ENUM(Ant::Behavior, {{Ant::Behavior::Default, 0},
                                             {Ant::Behavior::Conservative, 1},
                                             {Ant::Behavior::Randomized, 2},
                                             {Ant::Behavior::Bewitched, 3},
                                             {Ant::Behavior::ControlFree, 4}})

// Serialize Ant.
inline void to_json(json &j, const Ant &ant) {
    j = json{{"player", ant.get_player()},
             {"id", ant.get_id()},
             {"pos", {{"x", ant.get_x()}, {"y", ant.get_y()}}},
             {"hp", ant.get_hp()},
             // Use -1 to indicate that this is a new ant without any move
             {"move", ant.get_last_move()},
             {"level", ant.get_level()},
             {"age", ant.get_age()},
             {"status", ant.get_status()},
             {"behavior", ant.get_behavior()}};
}

// Add a Ant into the json object.
void Output::add_ant(const Ant &ant) { cur["ants"].push_back(ant); }

// Serialize DefensiveTower.
inline void to_json(json &j, const DefenseTower &tower) {
    j = json{{"player", tower.get_player()},
             {"id", tower.get_id()},
             {"pos", {{"x", tower.get_x()}, {"y", tower.get_y()}}},
             {"cd", tower.get_cd()}};
}

// Add a DefensiveTower into the json object.
void Output::add_tower(const DefenseTower &tower, int type,
                       const std::vector<int> &att_target) {
    json j = tower;
    // Add items
    j += json::object_t::value_type("type", type);
    if (!att_target.empty() && type != -1)
        j += json::object_t::value_type("attack", att_target);
    // Push into array
    cur["towers"].push_back(std::move(j));
}

// Add the Coin's into the json object.
void Output::add_coins(const Coin &player0, const Coin &player1) {
    cur["coins"] = json::array_t({player0.get_coin(), player1.get_coin()});
}

// Add the Camp's into the json object.
void Output::add_camps(const Headquarter &player0, const Headquarter &player1) {
    cur["camps"] = json::array_t({player0.get_hp(), player1.get_hp()});
    cur["speedLv"] =
        json::array_t({player0.get_cd_level(), player1.get_cd_level()});
    cur["anthpLv"] =
        json::array_t({player0.get_ant_level(), player1.get_ant_level()});
}
// Helpers for serialize a multi-dimensional std::array. Using TMP.

// Get element type of any container supporting begin().
// Ref:
// https://stackoverflow.com/questions/44521991/type-trait-to-get-element-type-of-stdarray-or-c-style-array
template <typename T>
using element_type_t =
    std::remove_reference_t<decltype(*std::begin(std::declval<T &>()))>;

// Check if it's an array. Compatible with std::array.
// Ref:
// https://stackoverflow.com/questions/40924276/why-does-stdis-array-return-false-for-stdarray
template <class T> struct is_array : std::is_array<T> {};

template <class T, std::size_t N>
struct is_array<std::array<T, N>> : std::true_type {};

template <typename T> constexpr bool is_array_v = is_array<T>::value;

// Check if it's a nested array.
template <typename ArrayType>
constexpr bool is_nested_array_v = is_array_v<element_type_t<ArrayType>>;

// Get the dimension of a nested array.
template <typename ArrayType> constexpr std::size_t get_dim() {
    if constexpr (is_nested_array_v<ArrayType>)
        return 1 + get_dim<element_type_t<ArrayType>>();
    else
        return 1;
}

template <typename ArrayType>
constexpr std::size_t array_dim_v = get_dim<ArrayType>();

// Serialize multi-dimensional std::array.
template <typename T, std::size_t N>
inline void to_json(json &j, const std::array<T, N> &arr) {
    using ArrayType = std::array<T, N>;
    if constexpr (array_dim_v < ArrayType >>
                  1) // Recur when dimension higher than 1 (at least 2)
    {
        j = json::array();
        for (auto it = arr.begin(); it != arr.end(); ++it) {
            int index = std::distance(arr.begin(), it);
            to_json(j[index], *it);
        }
    } else // 1-dim array can be directly serialized.
    {
        j = arr;
    }
}

// Serialize double.
inline void to_json(json &j, const double &num) { j = (double)num; }

// Add information of pheromone into the json object.
void Output::add_pheromone(const Pheromone &pheromone) {
    // It can be naturally serialized because to_json is implemented for
    // multi-dim arrays as above.
    // double phe[2][MAP_SIZE][MAP_SIZE];
    // for (int i = 0; i < 2; i++)
    //     for (int j = 0; j < MAP_SIZE; j++)
    //         for (int k = 0; k < MAP_SIZE; k++)
    //             phe[i][j][k] = pheromone[j][k][i];
    cur["pheromone"] = pheromone;
}
// Add information of operation
void Output::add_operation(const std::vector<Operation> *op) {
    round_msg["op0"] = op[0];
    round_msg["op1"] = op[1];
}
// Restore current data and get a new json object.
void Output::next_round() { cur = new_record(); }
void Output::save_seed(unsigned long long random_seed) {
    round_msg["seed"] = random_seed;
}
void Output::save_data() {
    round_msg["round_state"] = cur;
    // The live protocol still needs pheromone, but the full replay does not.
    // Dropping it here keeps long matches from retaining large round snapshots.
    round_msg["round_state"].erase("pheromone");
    data.push_back(round_msg);
    round_msg.clear();
}
// Add information of winner and message
void Output::add_winner(const int &winner, const std::string &msg) {
    cur["winner"] = winner;
    cur["message"] = msg;
}

void Output::add_error(const std::string &error) { cur["error"] = error; }
// Dump current data to a file.
void Output::dump_cur(const std::string &file_name) const {
    std::ofstream fout(file_name, std::ios::trunc);
    fout << std::setw(2) << cur << '\n';
    fout.close();
}

// Dump ALL data to a file.
void Output::dump_all(const std::string &file_name) const {
    std::ofstream fout(file_name);

    fout << std::setw(2) << data << '\n';

    fout.close();
}

json Output::get_cur() const { return cur; }

void Output::update_cur(const std::deque<DefenseTower> &defensive_towers) {
    cur["towers"].clear();

    for (auto tower : defensive_towers) {
        if (!tower.destroy())
            add_tower(tower, tower.get_type(), tower.get_attack());
    }
}
