#include "../include/ant.h"
#include "../include/map.h"

#include <algorithm>
#include <cassert>
// Create an ant.

const int hp_list[3] = {20, 25, 25};
namespace {
constexpr int SPECIAL_BEHAVIOR_DECAY_TURNS = 5;
constexpr int WORKER_TOWER_ATTACK_DAMAGE[3] = {1, 2, 4};
constexpr int COMBAT_TOWER_ATTACK_DAMAGE = 5;
constexpr int COMBAT_HP = 30;

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
} // namespace

Ant::Ant(int player, int id, int x, int y, int level, Kind kind)
    : player(player),
      id(id),             // Set player and id (may be generated automatically?)
      pos_x(x), pos_y(y), // Set initial position
      level(level),       // Set level
      hp(kind == Kind::Combat ? COMBAT_HP : hp_list[level]), // Set HP and its limit
      age(0),
      hp_limit(kind == Kind::Combat ? COMBAT_HP : hp_list[level]),
      trail_cells({Pos(x, y)}),
      last_move(NoMove),
      path_len_total(0),
      shield(0),
      evasion_control_free_on_break(false),
      defend(false),
      is_frozen(false),
      all_frozen(false),
      is_chosen(false),
      invincible(false),
      evasion(false),
      behavior(Behavior::Default),
      kind(kind),
      behavior_rounds(0),
      behavior_expiry(0),
      target_x(-1),
      target_y(-1),
      has_pending_behavior(false),
      pending_behavior(Behavior::Default)
{
    update_move_weights();
}

// Get the player to which the ant belong.
int Ant::get_player() const { return player; }

// Get the ant's id.
int Ant::get_id() const { return id; }

// Get the x coordinate of the ant's current position.
int Ant::get_x() const { return pos_x; }

// Get the y coordinate of the ant's current position.
int Ant::get_y() const { return pos_y; }

// Get the ant's HP.
int Ant::get_hp() const { return hp; }

// Get the ant's level.
int Ant::get_level() const { return level; }
// Get the HP limit of the ant.
int Ant::get_hp_limit() const { return hp_limit; }
// Get the age of the ant.
int Ant::get_age() const { return age; }

// Get the length of path
int Ant::get_path_len() const { return path_len_total; }

int Ant::get_last_move() const { return last_move; }

const std::vector<Pos> &Ant::get_trail_cells() const { return trail_cells; }

Ant::Behavior Ant::get_behavior() const { return behavior; }

Ant::Kind Ant::get_kind() const { return kind; }

bool Ant::is_control_immune() const { return behavior == Behavior::ControlFree; }

bool Ant::is_combat_ant() const { return kind == Kind::Combat; }

int Ant::get_tower_attack_damage() const {
    if (kind == Kind::Combat)
        return COMBAT_TOWER_ATTACK_DAMAGE;
    return WORKER_TOWER_ATTACK_DAMAGE[level];
}

bool Ant::should_self_destruct_on_tower_attack() const {
    return kind == Kind::Combat && hp * 2 < hp_limit;
}

void Ant::increase_age() { age++; }

void Ant::increase_behavior_rounds() { behavior_rounds++; }

void Ant::set_behavior(Behavior new_behavior, bool reset_rounds,
                       int expiry_rounds, bool force) {
    if (!force && is_control_immune() && new_behavior != Behavior::ControlFree)
        return;
    behavior = new_behavior;
    if (reset_rounds)
        behavior_rounds = 0;
    behavior_expiry =
        expiry_rounds >= 0 ? expiry_rounds : default_behavior_expiry(new_behavior);
    if (behavior != Behavior::Bewitched) {
        target_x = -1;
        target_y = -1;
    }
}

void Ant::update_move_weights() {
    if (kind == Kind::Combat) {
        move_weights.progress = 1.3;
        move_weights.pheromone = 0.05;
        move_weights.crowding = 0.15;
        move_weights.expected_damage = 1.1;
        move_weights.control_risk = 0.45;
        move_weights.tower_pull = 1.75;
        move_weights.effect_pull = 0.35;
        return;
    }
    move_weights.progress = 1.05;
    move_weights.pheromone = 0.15;
    move_weights.crowding = 0.4;
    move_weights.expected_damage = 2.0;
    move_weights.control_risk = 1.15;
    move_weights.tower_pull = 0.45;
    move_weights.effect_pull = 0.55;
}

void Ant::set_kind(Kind new_kind) {
    kind = new_kind;
    update_move_weights();
}

void Ant::set_bewitch_target(int x, int y) {
    target_x = x;
    target_y = y;
}

bool Ant::reached_target() const { return target_x == pos_x && target_y == pos_y; }

void Ant::set_pending_behavior_to(Behavior new_behavior) {
    has_pending_behavior = true;
    pending_behavior = new_behavior;
}

void Ant::clear_pending_behavior() { has_pending_behavior = false; }

void Ant::grant_evasion(int stacks, bool grant_control_free_on_deplete) {
    if (stacks <= 0)
        return;
    shield = std::max(shield, stacks);
    evasion = shield > 0;
    evasion_control_free_on_break =
        evasion_control_free_on_break || grant_control_free_on_deplete;
}

void Ant::add_evasion(int stacks, bool grant_control_free_on_deplete) {
    if (stacks <= 0)
        return;
    shield += stacks;
    evasion = shield > 0;
    evasion_control_free_on_break =
        evasion_control_free_on_break || grant_control_free_on_deplete;
}

void Ant::reset_backtrack() { last_move = NoMove; }

// Get the status of the ant.
Ant::Status Ant::get_status() const {
    if (hp <= 0)
        return Status::Fail;
    if (player && pos_x == PLAYER_0_BASE_CAMP_X &&
        pos_y == PLAYER_0_BASE_CAMP_Y)
        return Status::Success;
    if (!player && pos_x == PLAYER_1_BASE_CAMP_X &&
        pos_y == PLAYER_1_BASE_CAMP_Y)
        return Status::Success;
    if (kind != Kind::Combat && age > age_limit)
        return Status::TooOld;
    if (is_frozen || all_frozen)
        return Status::Frozen;
    return Status::Alive;
}

void Ant::set_hp_true(int change) { hp += change; }
// Change HP
void Ant::set_hp(int change) {
    if (change < 0 && shield > 0) {
        change = 0;
        shield--;
        evasion = shield > 0;
        if (shield == 0 && evasion_control_free_on_break &&
            !is_control_immune()) {
            evasion_control_free_on_break = false;
            set_behavior(Behavior::ControlFree);
        }
    } else if (defend && change < 0 && (-change) * 2 < hp_limit) {
        change = 0;
    }
    hp += change;
    if (hp > hp_limit)
        hp = hp_limit;
}

// Move the ant in specified direction.
// Note that the given direction should be valid (possible to reach),
// so it will NOT be checked.
void Ant::move(int direction) {
    const int d[2][6][2] = {
        {{0, 1}, {-1, 0}, {0, -1}, {1, -1}, {1, 0}, {1, 1}},
        {{-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, 0}, {0, 1}}};
    path_len_total++;
    if (direction == NoMove) {
        last_move = NoMove;
        return;
    }

    pos_x += d[pos_y % 2][direction][0];
    pos_y += d[pos_y % 2][direction][1];
    last_move = direction;
    trail_cells.emplace_back(pos_x, pos_y);
}

void Ant::teleport_to(int x, int y) {
    pos_x = x;
    pos_y = y;
    last_move = NoMove;
    trail_cells.emplace_back(pos_x, pos_y);
}
