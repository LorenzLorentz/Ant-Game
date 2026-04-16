#pragma once

#include "json.hpp"
#include "operation.h"
#include "output.h"
#include "map.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

class from_judger_init {
    std::vector<int> player_list;
    int player_num;
    json config;
    std::string replay;

  public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(from_judger_init, player_list,
                                                player_num, config, replay)

    int get_player_num() const { return player_num; }
    const json &get_config() const { return config; }
    const std::string &get_replay() const { return replay; }
    int get_AI_state(int player) const { return player_list[player]; }
};

class from_judger_round {
  private:
    int player;
    std::string content;
    int time;
    json from_player_json;
    std::string from_player_oj;

    void op_oj_to_json();
    void op_json_to_oj();

  public:
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(from_judger_round, player,
                                                content, time);

    int get_player() const { return player; }
    const std::string &get_content() const { return content; }
    int get_time() const { return time; }
    void transfer_op(bool is_ai);
    void set_operation_list(const std::vector<Operation> &operations);
    void send_operation(bool is_ai);
    std::vector<Operation> get_op_list();
};

class to_judger {
  private:
    json json_to_judger_web_player;
    std::string oj_to_judger;
    bool is_ai[2];
    std::vector<std::string> info_to_player;
    int info_size;
    int round = 0;
    std::vector<int> listen;
    int object;
    std::vector<int> player;
    // -1 is judger, 0 is player0, 1 is player1

  public:
    int get_round() const { return round; }
    std::vector<int> get_listen() const { return listen; }
    bool if_ai(int player) const { return is_ai[player]; }
    const std::vector<int> &get_player() const { return player; }
    const std::vector<std::string> &get_info_to_player() const {
        return info_to_player;
    }

    void init_player_state(int player, bool is_ai_);
    // set config
    void config_to_judger(bool if_ai);
    // send initial message to players
    using Pheromone = Map::Pheromone; // A multi-dim array
    void init_to_player(const unsigned long long &random_seed, const Pheromone &pheromone);
    void listen_player(int obj_player);
    void send_info_to_judger(const json &src, const int &state);
    void cur_json_to_oj(const json &src, const int &state);
    void set_json_to_web_player(const json &src);
};

class end_from_judger {
    std::string end_state;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(end_from_judger, end_state);
};

// transform int to 4 bytes
void int_to_bytes(int des);
// output string/json content
void output_info(int object, json info);
void output_info(int object, std::string info);

int str_to_num(std::string str);
