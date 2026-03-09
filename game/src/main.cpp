#include "../include/game.hpp"
#include "../include/operation.h"

#include <csignal>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

Game game;

// save replay when handle signal
void set_signal() {
    auto signal_handle = [](int sig) {
        std::cerr << "logic terminated by signal " << sig << '\n';
        game.dump_result(game.get_record_file());
        exit(sig);
    };
    signal(SIGINT, signal_handle);
    signal(SIGTERM, signal_handle);
}

int main(/*int argc, char *argv[]*/) {
    // redirect
    // FILE *err_out = freopen("err.out", "w", stderr);
    // if (err_out == nullptr)return 0;
    // setvbuf ( err_out , NULL , _IONBF , 0);

    // for test
    // std::ofstream fout;
    // fout.open("test.out");

    set_signal();
    game.init();
    // use p1.json & p2.json to read operations
    // use output.json to send message
    // use replay_file (in game class) to record the geme
    // clean json
    while (!game.is_ended()) {
        game.listen(0);
        if (!game.round_read_from_judger(0)) {
            break;
        }

        // auto op0 = read_operation("p1.json");
        // if (!game.apply_operation(op0, 0)) {
        //     game.set_AI_state_IO(0);
        //     break;
        // }

        game.listen(1);
        if (!game.round_read_from_judger(1)) {
            break;
        }

        // auto op1 = read_operation("p2.json");
        // if (!game.apply_operation(op1, 1)) {
        //     game.set_AI_state_IO(1);
        //     break;
        // }
        
        if( !game.next_round() )
            break;
        game.dump_round_state(/* "output.json" */);
    }
    // NOTICE !!
    // THE NEXT TWO FUNCTIONS ARE NOT SUPPORTED BY OLD JUDGER
    // game.request_end_state();
    // game.receive_end_state();

    /* at last, send end info &
       dump_all to replay file */
    game.send_end_info();

    return 0;
}
