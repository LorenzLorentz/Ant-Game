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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 1) 初始化：读取 K M
    int seat = 0;
    long long seed = 0;
    if (!(cin >> seat >> seed)) {
        return 0;
    }
    string dummy;
    getline(cin, dummy); // 读掉这一行剩余部分

    if (seat != 0 && seat != 1) {
        // 非法 seat，直接退出
        return 0;
    }

    while (true) {
        // 2) 读取一轮局面信息
        string line;
        if (!getline(cin, line)) {
            break; // EOF
        }
        // 跳过空行
        if (line.find_first_not_of(" \t\r\n") == string::npos) {
            continue;
        }

        int R = 0;
        {
            stringstream ss(line);
            if (!(ss >> R)) {
                // 非法 round 行，结束
                break;
            }
        }

        // N1: 防御塔数量
        if (!getline(cin, line)) break;
        int N1 = 0;
        {
            stringstream ss(line);
            if (!(ss >> N1)) break;
        }
        // 读 N1 行塔信息（本策略暂不使用，但按协议必须读）
        for (int i = 0; i < N1; ++i) {
            if (!getline(cin, line)) break;
            // id player x y type cd
            // 可以按需解析，这里直接略过
        }

        // N2: 蚂蚁数量
        if (!getline(cin, line)) break;
        int N2 = 0;
        {
            stringstream ss(line);
            if (!(ss >> N2)) break;
        }
        // 读 N2 行蚂蚁信息（同样暂不使用）
        for (int i = 0; i < N2; ++i) {
            if (!getline(cin, line)) break;
            // id player x y hp lv age state
        }

        // 金币行: G0 G1
        if (!getline(cin, line)) break;
        int G0 = 0, G1 = 0;
        {
            stringstream ss(line);
            if (!(ss >> G0 >> G1)) break;
        }
        int my_coin = (seat == 0 ? G0 : G1);

        // 基地血量行: HP0 HP1
        if (!getline(cin, line)) break;
        int HP0 = 0, HP1 = 0;
        {
            stringstream ss(line);
            if (!(ss >> HP0 >> HP1)) break;
        }

        // 3) 策略决策（等价于 Python 的 policy_from_site_state）

        vector<array<int,3>> ops; // 每条操作最多三个参数 [T x y] / [31] 等
        if (my_coin >= 50) {
            auto [bx, by] = base_pos(seat);
            const int OFF[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
            for (auto &d : OFF) {
                int x = bx + d[0];
                int y = by + d[1];
                if (0 <= x && x < 19 && 0 <= y && y < 19) {
                    // 11 x y 在 (x,y) 建塔，具体合法性由 C++ 游戏逻辑判定
                    ops.push_back({TOWER_BUILD, x, y});
                    break;
                }
            }
        }
        // 如果金币不足 50 或没找到合适位置，则 ops 为空，对应 N=0

        // 4) 按评测协议构造文本并发送
        // 文本格式：
        //   N
        //   11 x y
        //   ...
        ostringstream out;
        out << ops.size() << "\n";
        for (auto &op : ops) {
            if (op[0] == TOWER_BUILD) {
                out << op[0] << " " << op[1] << " " << op[2] << "\n";
            } else {
                // 如果以后扩展其它操作类型，在这里按需输出
                out << op[0] << "\n";
            }
        }
        string payload = out.str();
        send_packet(payload);
    }
    return 0;
}