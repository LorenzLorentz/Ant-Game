#include "../include/ant.h"
#include "../include/map.h"

#include <cassert>
// Create an ant.

const int hp_list[3] = {10, 25, 50};
Ant::Ant(int player, int id, int x, int y, int level)
    : player(player),
      id(id),             // Set player and id (may be generated automatically?)
      pos_x(x), pos_y(y), // Set initial position
      level(level),       // Set level
      hp(hp_list[level]), // Set HP and its limit
      age(0),
      hp_limit(hp_list[level]),
      trail_cells({Pos(x, y)}),
      last_move(NoMove),
      path_len_total(0),
      shield(0),
      defend(false),
      is_frozen(false),
      all_frozen(false),
      is_chosen(false),
      invincible(false),
      evasion(false),
      behavior(Behavior::Default),
      behavior_rounds(0),
      target_x(-1),
      target_y(-1),
      has_pending_behavior(false),
      pending_behavior(Behavior::Default)
{}

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

bool Ant::is_control_immune() const { return behavior == Behavior::ControlFree; }

void Ant::increase_age() { age++; }

void Ant::increase_behavior_rounds() { behavior_rounds++; }

void Ant::set_behavior(Behavior new_behavior, bool reset_rounds) {
    if (is_control_immune() && new_behavior != Behavior::ControlFree)
        return;
    behavior = new_behavior;
    if (reset_rounds)
        behavior_rounds = 0;
    if (behavior != Behavior::Bewitched) {
        target_x = -1;
        target_y = -1;
    }
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
    if (age > age_limit)
        return Status::TooOld;
    if (is_frozen || all_frozen)
        return Status::Frozen;
    return Status::Alive;
}

void Ant::set_hp_true(int change) { hp += change; }
// Change HP
void Ant::set_hp(int change) {
    if (shield > 0) {
        change = 0;
        shield--;
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
