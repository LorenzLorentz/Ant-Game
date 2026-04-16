#include "../include/comm_judger.h"
#include <cctype>
#include <cstdint>
#include <iomanip>

namespace {
std::uint32_t read_be_u32(const std::string &input) {
    std::uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
        value = (value << 8) + static_cast<unsigned char>(input[index]);
    }
    return value;
}

bool strip_length_prefix(const std::string &input, std::string &payload) {
    if (input.size() < 4) {
        return false;
    }
    std::uint32_t declared = read_be_u32(input);
    if (declared != input.size() - 4) {
        return false;
    }
    payload = input.substr(4);
    return true;
}

std::string normalize_ai_payload(const std::string &input) {
    std::string payload = input;
    std::string stripped;
    if (strip_length_prefix(payload, stripped)) {
        payload = std::move(stripped);
    }

    std::size_t start = 0;
    while (start < payload.size()) {
        unsigned char ch = static_cast<unsigned char>(payload[start]);
        if (std::isdigit(ch) || std::isspace(ch)) {
            break;
        }
        ++start;
    }
    if (start != 0 && start < payload.size()) {
        payload = payload.substr(start);
    }

    while (!payload.empty() &&
           static_cast<unsigned char>(payload.back()) == '\0') {
        payload.pop_back();
    }
    return payload;
}
} // namespace

// convert To_json object to json
inline void to_json(json &j, const to_judger &to_judger_) {
    j = json{{"state", to_judger_.get_round()},
             {"listen", to_judger_.get_listen()},
             {"player", to_judger_.get_player()},
             {"content", to_judger_.get_info_to_player()}};
}

// convert Pos to json
inline void to_json(json &j, const Pos &pos) {
    j = json{{"x", pos.x}, {"y", pos.y}};
}

void int_to_bytes(int des) {
    char tmp;
    for (int i = 3; i >= 0; i--) {
        tmp = (des >> (i * 8)) % (1 << 8);
        std::cout << tmp;
    }
}

void output_info(int object, json info) {
    size_t info_size = info.dump().size();
    int_to_bytes(info_size);
    int_to_bytes(object);
    std::cout << info;
    std::cout.flush();
}

void output_info(int object, std::string info) {
    size_t info_size = info.size();
    int_to_bytes(info_size);
    int_to_bytes(object);

    std::cout << info;
    std::cout.flush();
}

/* this function transfer operations from player
    into json | oj format and save operations */
void from_judger_round::transfer_op(bool is_ai) {
    if (!is_ai) {
        try {
            from_player_json = json::parse(content);
            op_json_to_oj();
        } catch (const std::exception &e) {
            std::cerr << "Operation from web_player is not a json\n";
            std::cerr << content << '\n';
            exit(0);
        }
    } else {
        std::string normalized = normalize_ai_payload(content);
        if (normalized != content) {
            content = normalized;
        }
        op_oj_to_json();
        from_player_oj = normalized;
    }
}

std::vector<Operation> from_judger_round::get_op_list() {
    try {
        std::vector<Operation> op_list = from_player_json;
    } catch (const std::exception &e) {
        std::cerr << "Get Operation List Error !\n";
        exit(0);
    }
    return from_player_json;
}

void from_judger_round::set_operation_list(
    const std::vector<Operation> &operations) {
    from_player_json = operations;
    op_json_to_oj();
}

/* if another player is ai, send oj operation; if
   another player is human, send json operation */
void from_judger_round::send_operation(bool is_ai) {
    int object = -1;
    if (player == 0) {
        object = 1;
    } else {
        object = 0;
    }

    if (is_ai) {
        output_info(object, from_player_oj);
    } else {
        output_info(object, from_player_json);
    }
}

void to_judger::init_player_state(int player, bool is_ai_) {
    is_ai[player] = is_ai_;
}

void to_judger::config_to_judger(bool if_ai) {
    object = -1;
    json j;
    if (if_ai)
        j = json{{"state", 0}, {"time", 10}, {"length", 2048}};
    else
        j = json{{"state", 0}, {"time", 300}, {"length", 2048}};
    output_info(object, j);
}

void to_judger::init_to_player(const unsigned long long &random_seed, const Pheromone &pheromone) {
    listen.clear();
    round = 1;
    player.clear();
    info_to_player.clear();

    for (int i = 0; i < 2; i++) {
        if (is_ai[i]) {
            player.push_back(i);
            info_to_player.push_back(std::to_string(i) + " " +
                                     std::to_string(random_seed) + "\n");
        } else {
            player.push_back(i);
            // double phe[2][19][19];
            // for (int i = 0; i < 2; i++)
            //     for (int j = 0; j < 19; j++)
            //         for (int k = 0; k < 19; k++)
            //             phe[i][j][k] = pheromone[j][k][i];
            json init_to_human = {{"player", i}, {"pheromone", pheromone}};
            info_to_player.push_back(init_to_human.dump());
        }
    }

    object = -1;
    json all_info = *this;
    output_info(object, all_info);
}

void to_judger::listen_player(int obj_player) {
    config_to_judger(is_ai[obj_player]);

    round++;
    listen.clear();
    listen.push_back(obj_player);
    player.clear();
    info_to_player.clear();

    object = -1;
    json all_info = *this;
    output_info(object, all_info);
}

void to_judger::send_info_to_judger(const json &src, const int &state) {
    listen.clear();
    player.clear();
    info_to_player.clear();

    cur_json_to_oj(src, state);

    //dump json to oj
    std::ofstream fout("mini_replay.txt", std::ios_base::app);
    fout << oj_to_judger;
    json phe = src["pheromone"];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < MAP_SIZE; i++) {
            for (int j = 0; j < MAP_SIZE; j++) {
                fout<< phe[k][i][j] << " ";
            }
            fout << std::endl;
        }
        // fout << std::endl;        
    }
    fout.close();

    // judge what player is : ai or human
    for (int i = 0; i < 2; i++) {
        if (is_ai[i]) {
            info_to_player.push_back(oj_to_judger);
            player.push_back(i);
        } else {
            info_to_player.push_back(json_to_judger_web_player.dump());
            player.push_back(i);
        }
    }

    object = -1;
    json all_info = *this;

    output_info(object, all_info);
}

void to_judger::cur_json_to_oj(const json &src, const int &state) {
    oj_to_judger.clear();

    oj_to_judger = std::to_string(state) + "\n";
    std::vector<json> json_list = src["towers"].get<json::array_t>();
    oj_to_judger += std::to_string(json_list.size());
    oj_to_judger += '\n';
    for (auto tower_info : json_list) {
        oj_to_judger = oj_to_judger +
                       std::to_string(tower_info["id"].get<int>()) + " " +
                       std::to_string(tower_info["player"].get<int>()) + " " +
                       std::to_string(tower_info["pos"]["x"].get<int>()) + " " +
                       std::to_string(tower_info["pos"]["y"].get<int>()) + " " +
                       std::to_string(tower_info["type"].get<int>()) + " " +
                       std::to_string(tower_info["cd"].get<int>()) + " " +
                       std::to_string(tower_info["hp"].get<int>()) + "\n";
    }
    json_list = src["ants"].get<json::array_t>();
    oj_to_judger += std::to_string(json_list.size());
    oj_to_judger += '\n';
    for (auto ant_info : json_list) {
        oj_to_judger = oj_to_judger +
                       std::to_string(ant_info["id"].get<int>()) + " " +
                       std::to_string(ant_info["player"].get<int>()) + " " +
                       std::to_string(ant_info["pos"]["x"].get<int>()) + " " +
                       std::to_string(ant_info["pos"]["y"].get<int>()) + " " +
                       std::to_string(ant_info["hp"].get<int>()) + " " +
                       std::to_string(ant_info["level"].get<int>()) + " " +
                       std::to_string(ant_info["age"].get<int>()) + " " +
                       std::to_string(ant_info["status"].get<int>()) + " " +
                       std::to_string(ant_info.value("behavior", 0)) + " " +
                       std::to_string(ant_info.value("kind", 0)) + "\n";
    }
    std::vector<int> coin = src["coins"].get<std::vector<int>>();
    oj_to_judger = oj_to_judger + std::to_string(coin[0]) + " " +
                   std::to_string(coin[1]) + "\n";

    std::vector<int> hp = src["camps"].get<std::vector<int>>();
    std::vector<int> speed_lv = src.value("speedLv", std::vector<int>{0, 0});
    std::vector<int> anthp_lv = src.value("anthpLv", std::vector<int>{0, 0});
    oj_to_judger = oj_to_judger + std::to_string(hp[0]) + " " +
                   std::to_string(hp[1]) + " " +
                   std::to_string(speed_lv[0]) + " " +
                   std::to_string(speed_lv[1]) + " " +
                   std::to_string(anthp_lv[0]) + " " +
                   std::to_string(anthp_lv[1]) + "\n";

    std::vector<json> cooldown_rows =
        src.value("weaponCooldowns", json::array()).get<json::array_t>();
    oj_to_judger += std::to_string(cooldown_rows.size()) + "\n";
    for (const auto &row : cooldown_rows) {
        std::vector<int> values = row.get<std::vector<int>>();
        for (size_t index = 0; index < values.size(); ++index) {
            if (index)
                oj_to_judger += " ";
            oj_to_judger += std::to_string(values[index]);
        }
        oj_to_judger += "\n";
    }

    std::vector<json> active_effects =
        src.value("activeEffects", json::array()).get<json::array_t>();
    oj_to_judger += std::to_string(active_effects.size()) + "\n";
    for (const auto &effect : active_effects) {
        oj_to_judger = oj_to_judger +
                       std::to_string(effect["type"].get<int>()) + " " +
                       std::to_string(effect["player"].get<int>()) + " " +
                       std::to_string(effect["x"].get<int>()) + " " +
                       std::to_string(effect["y"].get<int>()) + " " +
                       std::to_string(effect["duration"].get<int>()) + "\n";
    }
}

void to_judger::set_json_to_web_player(const json &src) {
    json_to_judger_web_player = src;
}

int str_to_num(std::string str) {
    int res = 0;
    for(auto s : str) {
        if(s < '0' || s > '9') {
            res = -1;
            break;
        }
        res = res * 10 + (s - '0');
    }
    return res;
}

void from_judger_round::op_oj_to_json() {
    // error operation
    from_player_json = json::array();
    Operation one_op = Operation(-1, -1, -1, -1, -1);
    from_player_json.push_back(std::move(one_op));

    std::vector<std::string> strlist;
    std::istringstream iss(content);
    std::string str;
    while (iss >> str) {
        strlist.push_back(str);
    }
    if(strlist.empty()) {
        return;
    }
    int size = str_to_num(strlist[0]);
    if(size < 0) {
        return;
    }
    int type = -1;
    int id = -1;
    int args = -1;
    int pos_x = -1;
    int pos_y = -1;
    from_player_json = json::array();
    unsigned int index = 1;
    for (int i = 0; i < size; i ++) {
        type = -1;
        id = -1;
        args = -1;
        pos_x = -1;
        pos_y = -1;
        if(index >= strlist.size()) {
            Operation one_op = Operation(type, id, args, pos_x, pos_y);
            from_player_json.push_back(std::move(one_op));
            continue;
        }
        
        if (strlist[index] == "11") {
            type = 11;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "12") {
            type = 12;
            id = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            args = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "13") {
            type = 13;
            id = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "21") {
            type = 21;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "22") {
            type = 22;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "23") {
            type = 23;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "24") {
            type = 24;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "31") {
            type = 31;
        } else if (strlist[index] == "32") {
            type = 32;
        }
        // debug operation
        else if (strlist[index] == "91") {
            type = 91;
            pos_x = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
            pos_y = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "92") {
            type = 92;
            id = ((++index) >= strlist.size()) ? -1 : str_to_num(strlist[index]);
        } else if (strlist[index] == "93") {
            type = 93;
        } else {
            std::cerr
                << "Operations of player contains undefined operation type!\n";
            std::cerr << "Undefined type : " << strlist[index] << std::endl;
        }

        Operation one_op = Operation(type, id, args, pos_x, pos_y);
        from_player_json.push_back(std::move(one_op));

        index ++;
    }
    if (index != strlist.size()) {
        Operation one_op = Operation(-1, -1, -1, -1, -1);
        from_player_json.push_back(std::move(one_op));
    }
}

void from_judger_round::op_json_to_oj() {
    from_player_oj.clear();

    from_player_oj += std::to_string(from_player_json.size());
    from_player_oj += "\n";

    for (auto op : from_player_json) {
        switch (op["type"].get<int>()) {
        case 11:
            from_player_oj = from_player_oj + "11" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 12:
            from_player_oj = from_player_oj + "12" + " " +
                             std::to_string(op["id"].get<int>()) + " " +
                             std::to_string(op["args"].get<int>()) + "\n";
            break;

        case 13:
            from_player_oj = from_player_oj + "13" + " " +
                             std::to_string(op["id"].get<int>()) + "\n";
            break;

        case 21:
            from_player_oj = from_player_oj + "21" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 22:
            from_player_oj = from_player_oj + "22" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 23:
            from_player_oj = from_player_oj + "23" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 24:
            from_player_oj = from_player_oj + "24" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 31:
            from_player_oj = from_player_oj + "31" + "\n";
            break;

        case 32:
            from_player_oj = from_player_oj + "32" + "\n";
            break;

        case 91:
            from_player_oj = from_player_oj + "91" + " " +
                             std::to_string(op["pos"]["x"].get<int>()) + " " +
                             std::to_string(op["pos"]["y"].get<int>()) + "\n";
            break;

        case 92:
            from_player_oj = from_player_oj + "92" + " " +
                             std::to_string(op["id"].get<int>()) + "\n";
            break;

        case 93:
            from_player_oj = from_player_oj + "93" + "\n";
            break;

        default:
            std::cerr
                << "Operations of player contains undefined operation type!\n";
            std::cerr << "Undefined type : " << op["type"].get<int>()
                      << std::endl;
            from_player_oj =
                from_player_oj + std::to_string(op["type"].get<int>()) + "\n";
            break;
        }
    }
}
