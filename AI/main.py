import sys
import struct
import json
import random


def log(msg):
    sys.stderr.write(f"[AI LOG] {msg}\n")
    sys.stderr.flush()

def receive_from_judger():
    # 评测机直接发原始内容，不封装，按行读取
    try:
        log("receive_from_judger")
        line = sys.stdin.readline()
        log(f"line: {line[:200]}...")
        if not line:
            log("no line")
            return None
        return json.loads(line)
    except Exception as e:
        log(f"receive_from_judger error: {e}")
        return None

def send_to_judger(result):
    # 4字节大端长度+正文
    try:
        data = (json.dumps(result) + "\n").encode("utf-8")
        header = struct.pack(">I", len(data))
        log(f"header: {header}")
        log(f"send_to_judger: {data}")
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except Exception as e:
        log(f"send_to_judger error: {e}")

def check_cell_valid(cells, x, y, player):
    """基于字典状态检查格子是否属于玩家且有足够兵力"""
    for cell in cells:
        pos, cell_player, cell_army = cell
        if pos == [x, y]:
            # 检查是否属于当前玩家且兵力大于1
            if cell_player == player and cell_army > 1:
                return True, cell_army
            return False, 0
    return False, 0

def random_action(state):
    player = state.get("Player", 0)
    generals = state.get("Generals", [])
    rest_move_step = state.get("RestMoveStep", [2, 2])[player] if "RestMoveStep" in state else 2
    cells = state.get("Cells", [])

    # 检查剩余移动步数
    if rest_move_step <= 0:
        return "8\n"

    main_general = None
    for g in generals:
        if g.get("Type", 1) == 1 and g.get("Player", -1) == player and g.get("Alive", 1):
            main_general = g
            break
    if not main_general:
        return "8\n"
    x, y = main_general.get("Position", [0, 0])

    # 查找主将格子的兵力
    valid, army = check_cell_valid(cells, x, y, player)
    if not valid or army <= 1:
        return "8\n"
    
    # 尝试从主将位置移动
    d = random.randint(0, 3)
    num = 1
    
    # 计算目标位置
    direction_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
    dx, dy = direction_offsets[d]
    new_x, new_y = x + dx, y + dy
    
    # 检查目标位置是否在范围内（假设棋盘是16x16，索引从0开始）
    if 0 <= new_x < 16 and 0 <= new_y < 16:
        return f"1 {x} {y} {d + 1} {num}\n8\n"
    
    # 如果主将位置移动失败，尝试随机选择其他己方格子
    max_tries = 10
    cnt = 0
    valid_move = False
    candidate_cells = []
    
    # 收集所有属于当前玩家且兵力大于1的格子
    for cell in cells:
        pos, cell_player, cell_army = cell
        if cell_player == player and cell_army > 1:
            candidate_cells.append((pos[0], pos[1], cell_army))
    
    if candidate_cells:
        # 随机选择一个候选格子
        x, y, army = random.choice(candidate_cells)
        d = random.randint(0, 3)
        dx, dy = direction_offsets[d]
        new_x, new_y = x + dx, y + dy
        
        if 0 <= new_x < 16 and 0 <= new_y < 16:
            return f"1 {x} {y} {d + 1} {num}\n8\n"
    
    # 如果都失败了，直接结束回合
    return "8\n"
def main():
    
    while True:
        state = receive_from_judger()
        player = state.get("Player", 0)
        log(f"player: {player}")
        command = random_action(state)
        result = {"player": player, "content": command}
        send_to_judger(result)

if __name__ == "__main__":
    main()