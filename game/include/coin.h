#ifndef __COIN_H__
#define __COIN_H__

#include "ant.h"
#include "item.h"
#include "building.h"
#include <tuple>

class Coin {
  private:
    // current coin
    int coin;
    // basic income each round
    int basic_income;
    // price of building new barrack
    int barrack_building_price;
    // price of building new tower
    int tower_building_price;
    // penalty if barracks > 1
    int penalty;

  public:
    // constructor
    Coin();
    // getter
    int get_coin() const;

    // set coin by change
    void set_coin(int change);

    // round income and penalty
    std::tuple<bool, int> basic_income_and_penalty()
        const; // basic round income return (!(income - penalty < 0 && coin <=
               // 0), income - penalty)

    // income from different sources
    void
    income_ant_kill(const Ant &); // income when killing one ant successfully
    void income_tower_destroy(const DefenseTower &, int tower_count); // income when destroying or downgrading one tower
    void income_ant_arrive(); // income when ant arriving at the camp
    // judge if coin is enough in following conditions
    bool isEnough_tower_build(int tower_count)
        const; // cost when building one tower; return if coin is enough
    bool isEnough_tower_upgrade(const DefenseTower &)
        const; // cost when upgrading one tower; return if coin is enough

    // cost due to different reasons, it will change coin
    void cost_tower_build(int tower_count);
    void cost_tower_upgrade(const DefenseTower &);

    bool isEnough_base_camp_upgrade(const int &) const ;
    void cost_base_camp_upgrade(const int &);

    bool isEnough_item_applied(ItemType item) const;
    void cost_item(ItemType item);
};

#endif
