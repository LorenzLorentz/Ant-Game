#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import struct
import traceback
import time
import subprocess
import threading
import queue
import random

# ==========================================
# 调试辅助
# ==========================================
def log(msg):
    sys.stderr.write(f"[log] {msg}\n")
# ==========================================
# 1. 核心通信层
# ==========================================

def receive_from_judger():
    log("Receiving from judger...")
    header = sys.stdin.buffer.read(4)
    if len(header) < 4:
        log("receive_from_judger: EOF or incomplete header")
        return b""
    length = struct.unpack(">I", header)[0]
    log(f"Expecting {length} bytes")
    data = sys.stdin.buffer.read(length)
    sys.stdin.buffer.flush()
    return data

def send_to_judger(data, target=-1):
    try:
        length = len(data)
        header = struct.pack(">Ii", length, target)
        sys.stdout.buffer.write(header)
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    except Exception:
        log(f"IO Error send: {traceback.format_exc()}")

# ==========================================
# 2. 协议封装层
# ==========================================

def receive_init_info():
    log("Receiving init info from judger...")
    data = receive_from_judger()
    if not data: 
        return {}
    return json.loads(data.decode("utf-8"))

def send_round_config(time_limit, length_limit):
    log(f"Sending round config: T={time_limit}")
    cfg = {"state": 0, "time": time_limit, "length": length_limit}
    send_to_judger(json.dumps(cfg).encode("utf-8"))

def send_round_info(state, listen, player, content):
    log(f"Sending round info: Round {state}, Listen {listen}")
    info = {
        "state": state,
        "listen": listen,
        "player": player,
        "content": content,
    }
    # According to protocol (developer.html), this is a judger-level message
    # (state/listen/player/content) so target must be -1 so judger parses
    # and forwards content to the appropriate AI(s)/watchers.
    send_to_judger(json.dumps(info).encode("utf-8"), -1)


def forward_raw_to_ai(player_idx, raw_content: bytes):
    """Send raw message body directly to a specific AI via judger (target=player_idx).

    Per protocol, judger will forward the raw bytes 'as-is' to the AI.
    raw_content must be bytes (no extra wrapper added here).
    """
    send_to_judger(raw_content, player_idx)

def receive_ai_info():
    data = receive_from_judger()
    if not data: 
        log("receive_ai_info got empty data, returning player -1")
        return {"player": -1, "content": "{}"}
    return json.loads(data.decode("utf-8"))

def send_game_end_info(end_info_str, end_state_str):
    log("Sending Game End Info...")
    info = {"state": -1, "end_info": end_info_str, "end_state": end_state_str}
    send_to_judger(json.dumps(info).encode("utf-8"), -1)

# ==========================================
# 3. 业务逻辑层
# ==========================================

error_map = ["RE", "TLE", "OLE"]

def write_end_info(gamestate):
    try:
        with open(gamestate.replay_file, "a") as f:
            f.write(str({
                "Round": gamestate.round,
                "Player": gamestate.winner,
                "Action": [9],
            }).replace("'", '"') + "\n")
    except Exception as e:
        log(f"Write end info failed: {e}")

# 处理 AI 操作
def read_ai_information_and_apply(gamestate, player, enemy_human, local_mode=False, ai_info_override=None):
    from logic.ai2logic import execute_single_command
    from logic.game_rules import is_game_over
    from logic.gamestate import update_round

    log(f"Waiting for AI {player} input...")
    ai_info = None

    # If we're running in local/self-play mode, generate commands with simple_ai
    if ai_info_override is not None:
        ai_info = ai_info_override
    elif local_mode:
        try:
            from logic.simple_ai import random_move_ai
            # simple_ai returns a list of command lists (e.g. [[1, x, y, dir, num], [8]])
            cmds = random_move_ai(gamestate.round, player, gamestate)
            # convert to the same textual format we expect from an external AI
            command_list_str = "\n".join(" ".join(map(str, c)) for c in cmds) + "\n"
            ai_info = {"player": player, "content": command_list_str}
            log(f"Local AI {player} generated commands: {command_list_str.strip()}")
        except Exception as e:
            log(f"Local AI generation failed: {e}")
            ai_info = {"player": -1, "content": "{}"}
    else:
        ai_info = receive_ai_info()
    
    # 1. 检查接收是否异常
    if ai_info.get("player", -1) != player:
        log(f"AI Error: Received message for player {ai_info.get('player', -1)}, expected {player}")
        return True, ""  # 忽略非本玩家的消息，继续等待
    if ai_info is None or ai_info.get("player", -1) == -1:
        log("AI Error: Player -1 received")
        gamestate.winner = 1 - player
        write_end_info(gamestate)
        
        # 尝试解析错误类型
        end_list = ["OK", "OK"]
        try:
            content_json = json.loads(ai_info["content"])
            p_idx = content_json.get("player", player)
            err_code = content_json.get("error", 0)
            end_list[p_idx] = error_map[err_code] if err_code < len(error_map) else "RE"
            end_info = {"0": p_idx, "1": 1 - p_idx} # 简单的胜负归属
        except:
            end_list[player] = "RE"
            end_info = {"0": 1-player, "1": player}
            
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    command_list_str = ai_info["content"]
    # log(f"AI Content: {command_list_str.strip()}")

    # 2. 解析指令
    command_list = []
    try:
        lines = command_list_str.strip().split('\n')
        log(f"Command List: {lines}")
        for line in lines:
            if not line.strip(): continue
            parsed_line = json.loads(line)["content"]
            log(f"Parsed Line: {parsed_line}")
            if not parsed_line: continue
            command_lines = parsed_line.strip().split('\n')
            log(f"Command Lines: {command_lines}")
            for command_line in command_lines:
                parts = list(map(int, command_line.split()))
                command_list.append(parts)
        log(f"Command List: {command_list}")
    except Exception as e:
        log(f"Command Parse Error: {e}")
        gamestate.winner = 1 - player
        write_end_info(gamestate)
        send_game_end_info(json.dumps({"0": player, "1": 1-player}), json.dumps(["IA", "OK"]))
        return False, ""
        
    # 3. 执行指令
    success = True
    for command in command_list:
        if len(command) == 0: continue
        if command[0] == 8: # 结束回合指令
            break
        
        log(f"Executing: {command}")
        success = execute_single_command(player, gamestate, command[0], command[1:])
        
        if not success:
            log("Command execution failed (Invalid Action)")
            gamestate.winner = 1 - player
            break
        
        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1: 
            log(f"Game Over triggered inside command loop. Winner: {gamestate.winner}")
            break

    # 4. 结算
    if gamestate.winner != -1:
        write_end_info(gamestate)
        end_list = ["OK", "OK"]
        time.sleep(1500)
        if not success: end_list[player] = "IA"
        end_info = {"0": 1 - gamestate.winner, "1": gamestate.winner}
        send_game_end_info(json.dumps(end_info), json.dumps(end_list))
        return False, ""

    # 5. 回合结束后的状态更新与广播 (此前缺失的部分)
    # 如果是后手(player 1)结束回合，通常意味着一个大回合结束，需要更新状态
    if player == 1:
        update_round(gamestate)
        
        # 读取最新的 Replay 行发送给评测机（用于前端展示）
        try:
            # 注意：这里假设 gamestate 内部逻辑或 update_round 已经往文件里写了一行新的状态
            # 如果原代码逻辑是 update_round 负责写文件，这里就读取最后一行
            # 为了安全起见，这里加上 try-except，防止文件读取失败导致崩溃
            with open(gamestate.replay_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    update_info = json.loads(last_line)
                    update_info["Player"] = player # 修正 player 字段
                    
                    replay_msg = str(update_info).replace("'", '"') + "\n"
                    # 发送给双方（如果有必要，或者只发给 Player）
                    # 原逻辑：发送给 player
                    send_to_judger(replay_msg.encode("utf-8"), player)
                    if enemy_human:
                        send_to_judger(replay_msg.encode("utf-8"), 1 - player)
        except Exception as e:
            log(f"Error reading/sending replay update: {e}")
            # 不因为显存/回放更新失败而终止游戏，除非非常关键
            pass

        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1:
            write_end_info(gamestate)
            send_game_end_info(json.dumps({"0": 1 - gamestate.winner, "1": gamestate.winner}), json.dumps(["OK", "OK"]))
            return False, ""

    return True, command_list_str

# 处理人类操作（直接修改游戏状态，不通过评测机）
def read_human_information_and_apply(gamestate, player, enemy_human):
    from logic.ai2logic import execute_single_command
    from logic.game_rules import is_game_over
    from logic.gamestate import update_round

    log(f"Waiting for Human {player} input...")
    
    # 从本地文件读取人类玩家命令（不通过评测机）
    # 命令文件格式：每行一个命令，格式为 "命令类型 参数1 参数2 ..."
    # 例如: "1 5 5 1 10" 表示移动军队
    #      "8" 表示结束回合
    command_file = f"human_player_{player}_commands.txt"
    command_list_str = ""
    
    try:
        # 尝试从文件读取命令
        with open(command_file, "r") as f:
            command_list_str = f.read().strip()
            # 读取后清空文件，避免重复执行
            with open(command_file, "w") as f2:
                f2.write("")
    except FileNotFoundError:
        log(f"Command file {command_file} not found, skipping turn")
        # 如果文件不存在，跳过这个回合（人类玩家还没有输入命令）
        return True, ""  # 继续游戏，但不执行任何操作
    except Exception as e:
        log(f"Failed to read command file: {e}")
        return True, ""  # 继续游戏，但不执行任何操作
    
    if not command_list_str:
        log(f"No commands for human player {player}, skipping turn")
        return True, ""  # 继续游戏，但不执行任何操作
    
    log(f"Human Player {player} Commands: {command_list_str.strip()}")

    # 2. 解析指令（直接命令格式，每行一个命令）
    # 命令格式：直接是命令字符串，每行一个命令
    # 例如: "1 5 5 1 10" 表示移动军队
    #      "8" 表示结束回合
    command_list = []
    try:
        lines = command_list_str.strip().split('\n')
        log(f"Command Lines: {lines}")
        
        for line in lines:
            if not line.strip(): continue
            # 跳过注释行
            if line.strip().startswith('#'):
                continue
            try:
                parts = list(map(int, line.split()))
                if len(parts) > 0:
                    command_list.append(parts)
            except ValueError as e:
                log(f"Failed to parse line as command: {line}, error: {e}")
                continue
        
        log(f"Parsed Command List: {command_list}")
    except Exception as e:
        log(f"Command Parse Error: {e}")
        # 解析错误不终止游戏，只是跳过这个回合
        return True, ""
        
    # 3. 执行指令
    success = True
    for command in command_list:
        if len(command) == 0: continue
        if command[0] == 8: # 结束回合指令
            break
        
        log(f"Executing: {command}")
        success = execute_single_command(player, gamestate, command[0], command[1:])
        
        if not success:
            log("Command execution failed (Invalid Action)")
            gamestate.winner = 1 - player
            break
        
        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1: 
            log(f"Game Over triggered inside command loop. Winner: {gamestate.winner}")
            break

    # 4. 检查游戏是否结束（直接修改游戏状态，不向评测机发送）
    if gamestate.winner != -1:
        write_end_info(gamestate)
        log(f"Game Over! Winner: {gamestate.winner}")
        # 不向评测机发送，直接返回
        return False, ""

    # 5. 回合结束后的状态更新（直接修改游戏状态）
    # 如果是后手(player 1)结束回合，通常意味着一个大回合结束，需要更新状态
    if player == 1:
        update_round(gamestate)
        log(f"Round updated after player {player} turn. New round: {gamestate.round}")
        
        # 直接检查游戏是否结束，不向评测机发送
        gamestate.winner = is_game_over(gamestate)
        if gamestate.winner != -1:
            write_end_info(gamestate)
            log(f"Game Over after round update! Winner: {gamestate.winner}")
            return False, ""

    return True, command_list_str 


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    gamestate = None
    try:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--selfplay', action='store_true')
        parser.add_argument('--spawn-bots', action='store_true', help='Spawn local bot_wrapper processes for AIs')
        parser.add_argument('--bot-path', type=str, default='bot_wrapper.py', help='Path to bot wrapper script')
        parser.add_argument('--bot-timeout', type=float, default=1.0, help='Per-turn time limit for bot responses (seconds)')
        args, unknown = parser.parse_known_args()

        spawn_bots = args.spawn_bots
        bot_path = args.bot_path
        bot_timeout = args.bot_timeout

        from logic.constant import *
        from logic.gamestate import GameState, init_generals, update_round
        
        gamestate = GameState()
        init_generals(gamestate)
        gamestate.coin = [40, 40]

        log("Wait for Init Info...")
        init_info = receive_init_info()
        # If we got no init info from a judger, run in local self-play mode.
        local_mode = False
        if not init_info:
            local_mode = True
            log("No init info received, running in local mode.")
        log(f"Init Info: {init_info}")

        gamestate.replay_file = init_info.get("replay", "replay.json")
        player_type = init_info.get("player_list", [1, 1, 0])
        
        # 写入初始 Replay
        init_json = gamestate.trans_state_to_init_json(-1)
        init_json["Round"] = 0
        with open(gamestate.replay_file, "w") as f:
            f.write(json.dumps(init_json) + "\n")

        # 发送 Config
        if player_type[0] == 1:
            send_round_config(1, 1024)
        else:
            send_round_config(180, 1024)

        # 广播地图
        json0 = gamestate.trans_state_to_init_json(0)
        json0["Round"] = 0
        json1 = gamestate.trans_state_to_init_json(1)
        json1["Round"] = 0
        send_round_info(1, [0], [0, 1], [json.dumps(json0)+"\n", json.dumps(json1)+"\n"])

        log("Game Loop Started...")
        # If spawn_bots requested, start two bot processes and reader threads
        bot_procs = [None, None]
        bot_queues = [queue.Queue(), queue.Queue()]
        bot_threads = [None, None]

        def bot_reader(proc, q):
            try:
                while True:
                    header = proc.stdout.buffer.read(4)
                    if not header or len(header) < 4:
                        break
                    length = struct.unpack('>I', header)[0]
                    data = proc.stdout.buffer.read(length)
                    if not data:
                        break
                    q.put(data)
            except Exception as e:
                # reader exits on error
                log(f"bot_reader exit: {e}")

        if spawn_bots:
            for i in (0, 1):
                try:
                    proc = subprocess.Popen(
                        [sys.executable, bot_path, '--player', str(i)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                    )
                    bot_procs[i] = proc
                    t = threading.Thread(target=bot_reader, args=(proc, bot_queues[i]), daemon=True)
                    t.start()
                    bot_threads[i] = t
                    log(f"Spawned bot {i} -> pid {proc.pid}")
                except Exception as e:
                    log(f"Failed to spawn bot {i}: {e}")
        
        player = 0
        state = 1
        game_continue = True
        
        while game_continue:
            log(f"--- Turn Start: Round {state}, Player {player}, Type {player_type[player]} ---")
            
            if player_type[player] == 1: # 本地 AI
                # If we spawned bot processes, forward the message to bot stdin and wait for its 4+n reply
                if spawn_bots and bot_procs[player] is not None:
                    # Prepare the content that would be sent to this player. We reuse the existing send_round_info flow
                    # but forward the specific content to the bot directly when we need to listen.
                    # For simplicity, we send a minimal state update: the last replay line (or initial state)
                    try:
                        with open(gamestate.replay_file, 'r') as f:
                            lines = f.readlines()
                            last_line = lines[-1] if lines else '{}\n'
                    except Exception:
                        last_line = '{}\n'

                    # forward raw text (judger would) to bot stdin
                    try:
                        bot_procs[player].stdin.write((last_line.strip() + '\n').encode('utf-8'))
                        bot_procs[player].stdin.flush()
                    except Exception as e:
                        log(f"Failed to send to bot {player}: {e}")

                    # wait for response from bot queue
                    ai_info = None
                    try:
                        data = bot_queues[player].get(timeout=bot_timeout)
                        try:
                            ai_info = json.loads(data.decode('utf-8'))
                        except Exception:
                            ai_info = {"player": -1, "content": "{}"}
                    except queue.Empty:
                        # timeout -> judge will report TLE for this player
                        log(f"Bot {player} timed out (TLE)")
                        ai_info = {"player": -1, "content": json.dumps({"player": player, "state": state, "error": 1, "error_log": "TLE"})}

                    game_continue, op_str = read_ai_information_and_apply(
                        gamestate, player, player_type[1-player] == 2, local_mode=False, ai_info_override=ai_info
                    )
                else:
                    game_continue, op_str = read_ai_information_and_apply(
                        gamestate, player, player_type[1-player] == 2, local_mode=local_mode
                    )
            elif player_type[player] == 2: # 网页/人类
                game_continue, op_str = read_human_information_and_apply(
                    gamestate, player, player_type[1-player] == 2
                ) 
            else:
                log(f"Unknown player type: {player_type[player]}")
                game_continue = False

            if not game_continue:
                log("Game continue flag became False. Exiting loop.")
                break
            
            player = 1 - player
            state += 1
            
            # 发送下一回合配置
            if player_type[player] == 1:
                send_round_config(1, 1024)
                # 生成当前游戏状态的JSON并发送给下一个玩家
                try:
                    state_json = gamestate.trans_state_to_init_json(player)
                    state_json["Round"] = gamestate.round
                    state_json["Player"] = player
                    state_json_str = json.dumps(state_json) + "\n"
                    send_round_info(state, [player], [player], [state_json_str])
                    log(f"Sent game state to player {player}: Round={gamestate.round}")
                except Exception as e:
                    log(f"Error generating/sending game state: {e}")
                    # 如果生成失败，尝试从replay文件读取
                    try:
                        with open(gamestate.replay_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                last_line = lines[-1].strip()
                                state_json_str = last_line + "\n"
                                send_round_info(state, [player], [player], [state_json_str])
                            else:
                                log("No replay data available")
                    except Exception as e2:
                        log(f"Error reading replay file: {e2}") 
            elif player_type[player] == 2:
                # 人类玩家：不向评测机发送，直接更新游戏状态
                # 游戏状态已经通过 replay_file 保存，人类玩家可以从文件读取
                log(f"Human player {player} turn. Game state saved to {gamestate.replay_file}")
                # 可选：将当前状态写入一个便于人类玩家读取的文件
                try:
                    state_json = gamestate.trans_state_to_init_json(player)
                    state_json["Round"] = gamestate.round
                    state_json["Player"] = player
                    # 将状态写入人类玩家可读的文件
                    state_file = f"human_player_{player}_state.json"
                    with open(state_file, "w") as f:
                        f.write(json.dumps(state_json, indent=2) + "\n")
                    log(f"Game state written to {state_file} for human player {player}")
                except Exception as e:
                    log(f"Error writing game state for human player: {e}")

    except Exception:
        sys.stderr.write("!!! FATAL ERROR !!!\n")
        sys.stderr.write(traceback.format_exc())
        if gamestate and hasattr(gamestate, 'replay_file'):
            try:
                with open(gamestate.replay_file, "a") as f:
                    f.write(traceback.format_exc())
            except: pass
        # 尝试告知 Judger 崩溃
        try:
            send_game_end_info("{}", json.dumps(["RE", "RE"]))
        except: pass
        sys.exit(1)