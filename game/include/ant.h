#ifndef __ANT_H__
#define __ANT_H__

#include "pos.h"
#include <vector>
class Ant {
  private:
    int player;
    int id;
    int pos_x, pos_y;
    int level;
    int hp;
    int age;
    int hp_limit;
    void update_move_weights();

  public:
    static constexpr int NoMove = -1;

    enum Kind {
        Worker,
        Combat,
    };

    enum Behavior {
        Default,
        Conservative,
        Randomized,
        Bewitched,
        ControlFree,
    };

    struct MoveWeights {
        double progress = 1.0;
        double pheromone = 0.3;
        double crowding = 0.35;
        double expected_damage = 1.15;
        double control_risk = 0.85;
        double tower_pull = 0.0;
        double effect_pull = 0.0;
    };

    std::vector<Pos> trail_cells;
    int last_move = NoMove;
    int path_len_total = 0;
    static const int age_limit = 64;
    int shield=0;
    bool evasion_control_free_on_break = false;
    bool defend=false;
    bool is_frozen = false;
    bool all_frozen = false;
    bool is_chosen = false;
    bool invincible = false;
    bool evasion = false;
    Behavior behavior = Behavior::Default;
    Kind kind = Kind::Worker;
    MoveWeights move_weights;
    int behavior_rounds = 0;
    int behavior_expiry = 0;
    int target_x = -1;
    int target_y = -1;
    bool has_pending_behavior = false;
    Behavior pending_behavior = Behavior::Default;
    enum Status {
        Alive,   // Still alive
        Success, // Reach the other camp
        Fail,    // No HP
        TooOld,  // Too old
        Frozen   // Forzen
    };

    Ant(int player, int id, int x, int y, int level, Kind kind = Kind::Worker);

    int get_player() const;
    int get_id() const;
    int get_x() const;
    int get_y() const;
    int get_hp() const;
    int get_hp_limit() const;
    int get_level() const;
    int get_age() const;
    int get_path_len() const;
    int get_last_move() const;
    const std::vector<Pos> &get_trail_cells() const;
    Behavior get_behavior() const;
    Kind get_kind() const;
    bool is_control_immune() const;
    bool is_combat_ant() const;
    int get_tower_attack_damage() const;
    bool should_self_destruct_on_tower_attack() const;

    void increase_age();
    void increase_behavior_rounds();
    void set_behavior(Behavior new_behavior, bool reset_rounds = true,
                      int expiry_rounds = -1, bool force = false);
    void set_kind(Kind new_kind);
    void set_bewitch_target(int x, int y);
    bool reached_target() const;
    void set_pending_behavior_to(Behavior new_behavior);
    void clear_pending_behavior();
    void grant_evasion(int stacks, bool grant_control_free_on_deplete = true);
    void add_evasion(int stacks, bool grant_control_free_on_deplete = true);
    void reset_backtrack();

    Status get_status() const;

    void set_hp(int change);
    void set_hp_true(int change);
    void move(int direction);
    void teleport_to(int x, int y);
};

#endif
