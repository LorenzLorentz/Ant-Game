#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iostream>

using namespace std;

static const int TOWER_BUILD = 11;

// 来自规则文档：基地方坐标
static const int PLAYER_0_BASE_CAMP_X = 2;
static const int PLAYER_0_BASE_CAMP_Y = 10 - 1; // SIDE_LENGTH-1 = 9
static const int PLAYER_1_BASE_CAMP_X = 2 * 10 - 1 - 3; // MAP_SIZE-3 = 19-3 = 16
static const int PLAYER_1_BASE_CAMP_Y = 10 - 1;

pair<int,int> base_pos(int seat) {
    if (seat == 0) return {PLAYER_0_BASE_CAMP_X, PLAYER_0_BASE_CAMP_Y};
    return {PLAYER_1_BASE_CAMP_X, PLAYER_1_BASE_CAMP_Y};
}

// 向 stdout 写入 4 字节大端长度 + payload
void send_packet(const string &payload) {
    uint32_t len = static_cast<uint32_t>(payload.size());
    unsigned char header[4];
    header[0] = (len >> 24) & 0xFF;
    header[1] = (len >> 16) & 0xFF;
    header[2] = (len >> 8) & 0xFF;
    header[3] = len & 0xFF;
    fwrite(header, 1, 4, stdout);
    fwrite(payload.data(), 1, payload.size(), stdout);
    fflush(stdout);
}

// helpers to read a full state packet
static bool read_state_lines(vector<string> &out_lines) {
    string line;
    // round
    if (!getline(cin, line))
        return false;
    // skip blank
    while (line.find_first_not_of(" \t\r\n") == string::npos) {
        if (!getline(cin, line))
            return false;
    }
    out_lines.push_back(line);
    int R;
    {
        stringstream ss(line);
        if (!(ss >> R))
            return false; // malformed
    }
    // N1 + towers
    if (!getline(cin, line))
        return false;
    out_lines.push_back(line);
    int N1;
    {
        stringstream ss(line);
        if (!(ss >> N1))
            return false;
    }
    for (int i = 0; i < N1; ++i) {
        if (!getline(cin, line))
            return false;
        out_lines.push_back(line);
    }
    // N2 + ants
    if (!getline(cin, line))
        return false;
    out_lines.push_back(line);
    int N2;
    {
        stringstream ss(line);
        if (!(ss >> N2))
            return false;
    }
    for (int i = 0; i < N2; ++i) {
        if (!getline(cin, line))
            return false;
        out_lines.push_back(line);
    }
    // coins
    if (!getline(cin, line))
        return false;
    out_lines.push_back(line);
    // camps hp
    if (!getline(cin, line))
        return false;
    out_lines.push_back(line);
    return true;
}

static void parse_state(const vector<string> &lines,
                        int &round,
                        int &G0, int &G1,
                        int &HP0, int &HP1) {
    auto it = lines.begin();
    {
        stringstream ss(*it);
        ss >> round;
    }
    ++it;
    int N1;
    {
        stringstream ss(*it);
        ss >> N1;
    }
    ++it;
    for (int i = 0; i < N1; ++i)
        ++it;
    int N2;
    {
        stringstream ss(*it);
        ss >> N2;
    }
    ++it;
    for (int i = 0; i < N2; ++i)
        ++it;
    {
        stringstream ss(*it);
        ss >> G0 >> G1;
    }
    ++it;
    {
        stringstream ss(*it);
        ss >> HP0 >> HP1;
    }
}

// read opponent ops (no framing)
static bool recv_ops(vector<array<int,3>> &ops) {
    string line;
    if (!getline(cin, line))
        return false;
    // skip blanks
    while (line.find_first_not_of(" \t\r\n") == string::npos) {
        if (!getline(cin, line))
            return false;
    }
    int N;
    {
        stringstream ss(line);
        if (!(ss >> N))
            return false;
    }
    for (int i = 0; i < N; ++i) {
        if (!getline(cin, line))
            return false;
        stringstream ss(line);
        array<int,3> op = {0,0,0};
        ss >> op[0];
        if (!(op[0] == TOWER_BUILD)) {
            // other ops have single int
        } else {
            ss >> op[1] >> op[2];
        }
        ops.push_back(op);
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // init
    int seat = 0;
    long long seed = 0;
    if (!(cin >> seat >> seed))
        return 0;
    string dummy;
    getline(cin, dummy);
    if (seat != 0 && seat != 1)
        return 0;

    vector<string> state_lines;
    int round = 0;
    int G0 = 0, G1 = 0, HP0 = 0, HP1 = 0;

    // we keep last parsed coin counts so policy can use them if needed
    int my_coin = 0;

    while (true) {
        if (seat == 0) {
            // decide using current coin
            vector<array<int,3>> ops;
            if (my_coin >= 50) {
                auto [bx, by] = base_pos(seat);
                const int OFF[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
                for (auto &d : OFF) {
                    int x = bx + d[0];
                    int y = by + d[1];
                    if (0 <= x && x < 19 && 0 <= y && y < 19) {
                        ops.push_back({TOWER_BUILD, x, y});
                        break;
                    }
                }
            }
            ostringstream out;
            out << ops.size() << "\n";
            for (auto &op : ops) {
                if (op[0] == TOWER_BUILD)
                    out << op[0] << " " << op[1] << " " << op[2] << "\n";
                else
                    out << op[0] << "\n";
            }
            send_packet(out.str());

            // receive opponent ops (ignored)
            vector<array<int,3>> dummy_ops;
            if (!recv_ops(dummy_ops))
                break;

            // now read the state that follows
            if (!read_state_lines(state_lines))
                break;
            for (auto &l : state_lines) cerr << "[state] " << l << "\n";
            parse_state(state_lines, round, G0, G1, HP0, HP1);
            my_coin = (seat == 0 ? G0 : G1);
            state_lines.clear();
        } else {
            // second player: first receive opponent ops
            vector<array<int,3>> dummy_ops;
            if (!recv_ops(dummy_ops))
                break;

            // decide now
            vector<array<int,3>> ops;
            if (my_coin >= 50) {
                auto [bx, by] = base_pos(seat);
                const int OFF[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
                for (auto &d : OFF) {
                    int x = bx + d[0];
                    int y = by + d[1];
                    if (0 <= x && x < 19 && 0 <= y && y < 19) {
                        ops.push_back({TOWER_BUILD, x, y});
                        break;
                    }
                }
            }
            ostringstream out;
            out << ops.size() << "\n";
            for (auto &op : ops) {
                if (op[0] == TOWER_BUILD)
                    out << op[0] << " " << op[1] << " " << op[2] << "\n";
                else
                    out << op[0] << "\n";
            }
            send_packet(out.str());

            // then read the following state
            if (!read_state_lines(state_lines))
                break;
            for (auto &l : state_lines) cerr << "[state] " << l << "\n";
            parse_state(state_lines, round, G0, G1, HP0, HP1);
            my_coin = (seat == 0 ? G0 : G1);
            state_lines.clear();
        }
    }
    return 0;
}