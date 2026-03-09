#pragma once

#include "ant.h"
#include "building.h"
#include "coin.h"
#include "json.hpp"
#include "map.h"
#include "operation.h"
#include <array>
#include <deque>
#include <string>
#include <vector>

// Alias for multi-dim array based on nested std::array.

class Output {
  private:
    using json = nlohmann::json;
    std::vector<json> data;
    json cur;
    json round_msg;
    json mini_msg;
    // Return an empty record for cur.
    static json new_record();

  public:
    // Constructor
    Output();

    // Write in different kinds of data.
    void add_ant(const Ant &ant);
    void add_tower(const DefenseTower &tower,
                   int type, const std::vector<int>& att_target); // type is the type of operation

    void add_coins(const Coin &player0, const Coin &player1);
    void add_camps(const Headquarter &player0, const Headquarter &player1);
    void add_operation(const std::vector<Operation> *op);
    void add_winner(const int &, const std::string &);
    void add_error(const std::string& error);
    using Pheromone = Map::Pheromone; // A multi-dim array
    void add_pheromone(const Pheromone &pheromone);
    // Create a new json object to write in.
    void next_round();
    void save_seed(unsigned long long random_seed);
    void save_data();
    // Dump to file.
    void dump_cur(const std::string &file_name) const;
    void dump_all(const std::string &file_name) const;

    json get_cur() const;
    void update_cur(const std::deque<DefenseTower> &defensive_towers);
};
