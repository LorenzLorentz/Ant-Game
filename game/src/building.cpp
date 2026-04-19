#include "../include/building.h"
#include <cmath>
#include <algorithm>


const int cd_num[3] = {9, 4, 7};
const int cd_den[3] = {2, 1, 2};
// judge whether ant can be generated
bool Headquarter::create_new_ant(int round) {
    if (round == 0) return true;
    return (round * cd_den[cd_level]) / cd_num[cd_level] >
           ((round - 1) * cd_den[cd_level]) / cd_num[cd_level];
}
// current cd
int Headquarter::get_cd_level() const { return cd_level; }
// ants'hp_limit this round
int Headquarter::get_ant_level() const { return ant_level; }
// upgrade ant hp
bool Headquarter::ant_upgrade() {
    if (ant_level == 2) return false;
    ant_level ++;
    return true;
}
bool Headquarter::barrack_upgrade() {
    if (cd_level == 2) return false;
    cd_level ++;
    return true;
}

Ant *DefenseTower::find_attack_target(std::vector<Ant> &ants) {
    int m = 30;
    Ant *candidate = nullptr;
    for (auto &ant : ants) {
        if (tower_type == TowerType::Double && 
            std::find(attacked_ants.begin(), attacked_ants.end(), ant.get_id()) != attacked_ants.end()
        ) {
            continue;
        }
        if (get_player() != ant.get_player() && ant.get_hp() > 0) {
            int dist = distance(ant.get_x(), ant.get_y(), get_x(), get_y());
            if (dist <= get_range()) {
                if (dist < m) {
                    m = dist;
                    candidate = &ant;
                }
            }
        }
    }
    return candidate;
}

Ant *DefenseTower::attack1(std::vector<Ant> &ants, Ant first_attack_ant) {
    int m = 500;
    int flag = 0;

    Ant *attacked_ant = nullptr;
    if ((abs(get_x() - first_attack_ant.get_x()) *
             abs(get_x() - first_attack_ant.get_x()) +
         abs(get_y() - first_attack_ant.get_y()) *
             abs(get_y() - first_attack_ant.get_y())) <= range * range) {
        attacked_ant = &first_attack_ant;
        flag = 1;
    }
    if (flag == 0) {
        for (auto &ant : ants) {
            if (get_player() != ant.get_player() && ant.get_hp() > 0) {
                if (distance(ant.get_x(), ant.get_y(), get_x(), get_y()) <=
                    get_range()) {
                    if (abs(get_x() - ant.get_x()) +
                            abs(get_y() - ant.get_y()) <
                        m) {
                        m = abs(get_x() - ant.get_x()) +
                            abs(get_y() - ant.get_y());
                        attack_pos_x = ant.get_x();
                        attack_pos_y = ant.get_y();
                        attacked_ant = &ant;
                    } else if (abs(get_x() - ant.get_x()) +
                                   abs(get_y() - ant.get_y()) ==
                               m) {
                        if (ant.get_path_len() > attacked_ant->get_path_len()) {
                            attack_pos_x = ant.get_x();
                            attack_pos_y = ant.get_y();
                            attacked_ant = &ant;
                        }
                    }
                }
            }
        }
    }
    return attacked_ant;
}

Ant *DefenseTower::attack2(std::vector<Ant> &ants) {
    int m = 500;
    // int flag=0;

    Ant *attacked_ant = nullptr;
    for (auto &ant : ants) {
        if (get_player() != ant.get_player() && (ant.get_hp() > 0) &&
            (!ant.is_chosen)) {
            if (distance(ant.get_x(), ant.get_y(), get_x(), get_y()) <=
                get_range()) {
                if (abs(get_x() - ant.get_x()) + abs(get_y() - ant.get_y()) <
                    m) {
                    m = abs(get_x() - ant.get_x()) + abs(get_y() - ant.get_y());
                    attack_pos_x = ant.get_x();
                    attack_pos_y = ant.get_y();
                    attacked_ant = &ant;
                } else if (abs(get_x() - ant.get_x()) +
                               abs(get_y() - ant.get_y()) ==
                           m) {
                    if (ant.get_path_len() > attacked_ant->get_path_len()) {
                        attack_pos_x = ant.get_x();
                        attack_pos_y = ant.get_y();
                        attacked_ant = &ant;
                    }
                }
            }
        }
    }
    return attacked_ant;
}
// check the type of new tower
bool DefenseTower::upgrade_type_check(int new_type) const {
    try {
        auto t = TowerType(new_type);
        return TowerType(t / 10) == tower_type;
    } catch (const std::exception &e) {
        return false;
    }
}
// Get the downgrade type of tower
TowerType DefenseTower::tower_downgrade_type() const {
    return TowerType(tower_type / 10);
}

void DefenseTower::set_stats_for_type(TowerType tower_type_) {
    damage = 5;
    spd = 2;
    range = 1;
    hp_limit = 10;
    switch (tower_type_) {
    case TowerType::Basic:
        break;
    case TowerType::Heavy:
        damage = 12;
        spd = 2;
        range = 1;
        hp_limit = 15;
        break;
    case TowerType::Quick:
        damage = 6;
        spd = 1;
        range = 1;
        hp_limit = 15;
        break;
    case TowerType::Mortar:
        damage = 12;
        spd = 4;
        range = 2;
        hp_limit = 15;
        break;
    case TowerType::Producer:
        damage = 0;
        spd = 4;
        range = 0;
        hp_limit = 15;
        break;
    case TowerType::HeavyPlus:
        damage = 24;
        spd = 2;
        range = 1;
        hp_limit = 15;
        break;
    case TowerType::Ice:
        damage = 12;
        spd = 2;
        range = 2;
        hp_limit = 15;
        break;
    case TowerType::Bewitch:
        damage = 14;
        spd = 2;
        range = 2;
        hp_limit = 15;
        break;
    case TowerType::QuickPlus:
        damage = 6;
        spd = 0.5;
        range = 1;
        hp_limit = 15;
        break;
    case TowerType::Double:
        damage = 6;
        spd = 2;
        range = 3;
        hp_limit = 15;
        break;
    case TowerType::Sniper:
        damage = 10;
        spd = 2;
        range = 4;
        hp_limit = 15;
        break;
    case TowerType::MortarPlus:
        damage = 18;
        spd = 4;
        range = 2;
        hp_limit = 15;
        break;
    case TowerType::Pulse:
        damage = 14;
        spd = 4;
        range = 2;
        hp_limit = 15;
        break;
    case TowerType::Missile:
        damage = 18;
        spd = 6;
        range = 3;
        hp_limit = 15;
        break;
    case TowerType::ProducerFast:
        damage = 0;
        spd = 2;
        range = 0;
        hp_limit = 15;
        break;
    case TowerType::ProducerSiege:
        damage = 0;
        spd = 4;
        range = 0;
        hp_limit = 15;
        break;
    case TowerType::ProducerMedic:
        damage = 0;
        spd = 4;
        range = 0;
        hp_limit = 15;
        break;
    default:
        break;
    }
}
// upgrade the tower
bool DefenseTower::upgrade(TowerType tower_type_) {
    round = 0;
    level++;
    tower_type = tower_type_;
    set_stats_for_type(tower_type);
    hp = hp_limit;
    return true;
}

// downgrade the tower
bool DefenseTower::downgrade(TowerType tower_type_) {
    round = 0;
    int previous_hp = hp;
    int previous_hp_limit = hp_limit;
    level--;
    tower_type = tower_type_;
    set_stats_for_type(tower_type);
    hp = previous_hp_limit > 0
             ? std::max(1, (previous_hp * hp_limit + previous_hp_limit - 1) /
                               previous_hp_limit)
             : hp_limit;
    return true;
}

void DefenseTower::round_damage(std::vector<Ant> &ants, int x, int y,
                                int range) {
    for (auto &ant : ants) {
        if (get_player() != ant.get_player()) {
            if (distance(ant.get_x(), ant.get_y(), x, y) <= range) {
                ant.set_hp(-get_damage());
                add_attacked_ants(ant.get_id());
            }
        }
    }
}

int DefenseTower::distance(int x0, int y0, int x1, int y1) {
    int dy = abs(y0 - y1);
    int dx;
    if (abs(y0 - y1) % 2) {
        if (x0 > x1)
            dx = std::max(0, abs(x0 - x1) - abs(y0 - y1) / 2 - (y0 % 2));
        else
            dx = std::max(0, abs(x0 - x1) - abs(y0 - y1) / 2 - (1 - (y0 % 2)));
    } else
        dx = std::max(0, abs(x0 - x1) - abs(y0 - y1) / 2);

    return dx + dy;
}

void DefenseTower::set_changed_this_round() {
    changed = true;
}

void DefenseTower::set_unchanged_before_another_round() {
    changed = false;
    attacked_ants.clear();
}

void DefenseTower::add_attacked_ants(int id) {
    attacked_ants.push_back(id);
}

bool DefenseTower::is_producer() const {
    return tower_type == TowerType::Producer ||
           tower_type == TowerType::ProducerFast ||
           tower_type == TowerType::ProducerSiege ||
           tower_type == TowerType::ProducerMedic;
}

int DefenseTower::get_spawn_interval() const {
    switch (tower_type) {
    case TowerType::Producer:
        return 10;
    case TowerType::ProducerFast:
        return 8;
    case TowerType::ProducerSiege:
    case TowerType::ProducerMedic:
        return 10;
    default:
        return 0;
    }
}

int DefenseTower::get_support_interval() const {
    return tower_type == TowerType::ProducerMedic ? 4 : 0;
}

int DefenseTower::get_support_range() const {
    return 0;
}

double DefenseTower::get_siege_spawn_chance() const {
    return tower_type == TowerType::ProducerSiege ? 0.25 : 0.0;
}

int DefenseTower::get_heal_amount() const {
    return 0;
}

bool DefenseTower::take_damage(int amount) {
    if (amount <= 0)
        return false;
    hp -= amount;
    return hp <= 0;
}
