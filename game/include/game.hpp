#ifndef __GAME_H__
#define __GAME_H__

#include "ant.h"
#include "item.h"
#include "building.h"
#include "comm_judger.h"
#include "map.h"
#include "operation.h"
#include "output.h"
#include "player.h"
#include <deque>
#include <random>
#include <tuple>
#include <vector>

#define MAX_ROUND 512

class Game {
  private:
    bool is_end;
    int winner;
    int round;
    const std::string mini_replay = "mini_replay.txt";
    int ant_id = 0;
    int barrack_id = 0;
    int tower_id = 0;
    std::string err_msg = "";

    // state of AI
    enum AI_state {
        OK,
        INITIAL_ERROR,
        RUN_ERROR,
        TIMEOUT_ERROR,
        OUTPUT_LIMIT,
        ILLEGAL_OPERATION,
        HUMAN_PLAYER
    } state[2];

    std::string record_file;
    from_judger_round judger_round_info;
    to_judger output_to_judger;
    unsigned long long random_seed;
    unsigned long long rng_state = 0;
    Map map;
    // player[2]
    Player player0, player1;
    Headquarter base_camp0, base_camp1;
    std::vector<Operation> op[2];
    std::vector<Item> item[2];
    std::deque<DefenseTower> defensive_towers;
    std::vector<Ant> ants;

    Output output;

    void attack_ants();   // defensive towers attack ants
    void move_ants();     // get direction and move ants
    bool manage_ants();   // update game_data_output & manage ants by status
    void generate_ants(); // generate new ants
    void increase_ant_age();
    void update_items();      // update duration of item
    void update_coin();      // update coin by basic income and penalty
    void update_pheromone(); // update pheromone for each ant
    bool judge_base_camp();  // judge winner by base_camps' hp
    void judge_winner(); // judge winner when round is no less than 512
    unsigned long long next_random();
    double random_float();
    int random_index(int bound);
    int choose_ant_move(const Ant &ant);
    bool ant_can_walk_to(int x, int y) const;
    double crowding_penalty(const Ant &ant, int x, int y) const;
    void teleport_ants();
    void drift_items();
    std::pair<int, int> random_own_half_target(int player);
    bool ant_in_own_half(const Ant &ant) const;
    void apply_control(Ant &ant, Ant::Behavior behavior,
                       const std::pair<int, int> *target = nullptr);
    void maybe_control_free(Ant &ant, bool was_active, bool is_active);
    void prepare_ants_for_attack();
    void damage_ant_by_tower(DefenseTower &tower, Ant &ant);

  public:
    Game();
    ~Game();

    std::string get_record_file();
    void set_AI_state_IO(int player);
    void init();
    bool is_ended();
    bool next_round();

    // bool is_operation_valid(const OperationSet& op) const;
    bool apply_operation(const std::vector<Operation> &op_list,
                         int player, std::string& err_msg); // 进行选手在回合的操作
    void dump_round_state(/* const std::string& filename */);
    void
    dump_last_round(/* const std::string& filename */ const std::string &msg);
    void request_end_state();
    void receive_end_state();
    void send_end_info();
    void dump_result(const std::string &filename);
    // void dump_mini_replay(const std::string &filename);
    // void show(int t){map.show(t);}
    //  read message from judger
    template <typename T> void read_from_judger(T &des);
    void listen(int player);
    bool round_read_from_judger(int player);
};

#endif
