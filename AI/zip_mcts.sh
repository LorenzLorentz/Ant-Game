#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
rm -rf submissions/mcts_temp
rm -f submissions/mcts.zip
mkdir -p submissions/mcts_temp

cp AI/main.py submissions/mcts_temp/main.py
cp AI/ai_mcts.py submissions/mcts_temp/ai.py
# 【救命的关键】：把评测机点名要的通信协议文件塞进去！
cp AI/protocol.py submissions/mcts_temp/ 2>/dev/null || true
cp AI/common.py submissions/mcts_temp/ 2>/dev/null || true
cp -r SDK submissions/mcts_temp/

# Package the trained checkpoint under the name the agent looks for first.
if [ -f "AI/ai_mcts_model.npz" ]; then
    cp AI/ai_mcts_model.npz submissions/mcts_temp/
elif [ -f "checkpoints/ai_mcts_latest.npz" ]; then
    cp checkpoints/ai_mcts_latest.npz submissions/mcts_temp/ai_mcts_model.npz
fi

echo "numpy" > submissions/mcts_temp/requirements.txt

cd submissions/mcts_temp
zip -r ../mcts.zip ./*
cd ../..
rm -rf submissions/mcts_temp

echo "🎉 终极补漏版打包完成！请提交 submissions/mcts.zip 文件。"
