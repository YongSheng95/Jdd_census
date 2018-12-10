# Jdd_census
京东2018 JDD比赛: [人口动态普查与预测](https://jdder.jd.com/index/jddDetail?matchId=3dca1a91ad2a4a6da201f125ede9601a)

#base_line
- base_line只使用了3个特征，year,month,day,得分0.1425
- 在base_line基础上，进行了改进，加入了三个特征，前一天的dwell,flow_in, flow_out,得分不高

#lstm
- 考虑到此题可能会是序列预测，因此采用了LSTM
- 特征为dwell,flow_in, flow_out，没有加入year,month,day
- 得分反而下降

#思考
- transition文件还没有使用，之前想过在训练模型阶段使用，但是没法在预测阶段构造数据，因而没法使用
- 很难构造特征，方向不是很明确
- 最后放弃了，等待大佬们开源吧

