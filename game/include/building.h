#ifndef __BUILDING_H__
#define __BUILDING_H__

#include "ant.h"
#include <iostream>

class Building {
  private:
    int pos_x;
    int pos_y;
    int player; // 0 1
    int id;
    int type;

    bool _destroy = false;
    // 0 is Headquarter,
    // 1 is DefenseTower,
    // 2 is Barrack
  public:
    Building(){};
    Building(int x, int y, int player, int id, int type)
        : pos_x(x), pos_y(y), player(player), id(id), type(type){};
    int get_id() const { return id; }
    int get_player() const { return player; }
    int get_x() const { return pos_x; }
    int get_y() const { return pos_y; }
    bool set_destroy() {
        _destroy = true;
        return true;
    }
    bool destroy() const { return _destroy; }

    virtual ~Building(){};
};
// Tower type
enum TowerType {
    Basic = 0,
    Heavy = 1,
    HeavyPlus = 11,
    Ice = 12,
    Bewitch = 13,
    Quick = 2,
    QuickPlus = 21,
    Double = 22,
    Sniper = 23,
    Mortar = 3,
    MortarPlus = 31,
    Pulse = 32,
    Missile = 33,
    Producer = 4,
    ProducerFast = 41,
    ProducerSiege = 42,
    ProducerMedic = 43,
};

class DefenseTower : public Building {
  private:
    int level;
    TowerType tower_type;
    int damage = 5;
    double spd = 2;
    int range = 1;
    int hp = 10;
    int hp_limit = 10;
    int attack_pos_x = 0, attack_pos_y = 0;
    bool changed = false;
    std::vector<int> attacked_ants;
    void set_stats_for_type(TowerType tower_type_);

  public:
    double multiple = 1.0;
    int round = 0;

    DefenseTower(int x, int y, int player, int id, int type)
        : Building(x, y, player, id, type), level(0),
          tower_type(TowerType::Basic), attack_pos_x(x), attack_pos_y(y) {
        set_stats_for_type(TowerType::Basic);
        hp = hp_limit;
    }
    bool upgrade_type_check(int new_type) const;
    TowerType tower_downgrade_type() const;
    bool upgrade(TowerType tower_type_);
    int get_damage() const { return (int)(damage * multiple); }
    int get_spd() const { return spd; }
    int get_cd() const {
        if (is_producer())
            return std::max(get_spawn_interval() - round, 0);
        if (spd < 1)
            return 0;
        return std::max((int)(spd - round), 0);
    }
    int get_level() const { return level; }
    TowerType get_type() const { return tower_type; }
    int get_range() const { return range; }
    int get_hp() const { return hp; }
    int get_hp_limit() const { return hp_limit; }
    bool is_producer() const;
    int get_spawn_interval() const;
    int get_support_interval() const;
    int get_support_range() const;
    double get_siege_spawn_chance() const;
    int get_heal_amount() const;
    bool downgrade(TowerType tower_type_);
    bool is_changed() const {return changed;}
    const std::vector<int>& get_attack() const {return attacked_ants;}
    bool take_damage(int amount);

    virtual ~DefenseTower(){};
    Ant *find_attack_target(std::vector<Ant> &ants);
    Ant *attack1(std::vector<Ant> &ants, Ant first_attack_ant);
    Ant *attack2(std::vector<Ant> &ants);
    void round_damage(std::vector<Ant> &ants, int x, int y, int range);
    int distance(int x0, int y0, int x1, int y1);

    void set_changed_this_round();
    void set_unchanged_before_another_round();
    void add_attacked_ants(int id);
};

class Headquarter : public Building {
  private:
    int hp;
    int cd_level = 0;
    int ant_level = 0;
  public:
    Headquarter(){};
    Headquarter(int x, int y, int player, int id, int type, int hp)
        : Building(x, y, player, id, type), hp(hp){};
    int get_hp() const { return hp; }
    void set_hp(int change) { hp += change; }
    bool create_new_ant(int round);
    int get_cd_level() const;
    int get_ant_level() const;
    bool ant_upgrade();
    bool barrack_upgrade();
    virtual ~Headquarter(){};
};

#endif
