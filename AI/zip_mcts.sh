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

# 【打包 Numpy 权重文件】
if [ -f "model_weights.npz" ]; then
    cp model_weights.npz submissions/mcts_temp/
fi

echo "numpy" > submissions/mcts_temp/requirements.txt

cd submissions/mcts_temp
zip -r ../mcts.zip ./*
cd ../..
rm -rf submissions/mcts_temp

echo "🎉 终极补漏版打包完成！请提交 submissions/mcts.zip 文件。"