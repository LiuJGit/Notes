"""
LSTM 初体验
"""
import torch

#单向lstm
def single_lstm():
    batch_size = 10
    seq_len = 21
    embedding_dim = 30 # 输入数据的维度
    word_vocab = 100
    hidden_size = 18 # 隐变量的维度
    num_layer = 2

    #准备输入数据
    input = torch.randint(low=0,high=word_vocab,size=[batch_size,seq_len])
    #embedding
    embedding = torch.nn.Embedding(word_vocab,embedding_dim)
    embed = embedding(input)

    #实例化LSTM模型
    #batch_first: False[default]-->[seq_len,batch_size,feature];True-->[batch_size,seq_len,feature]
    lstm = torch.nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,
                         num_layers=num_layer,batch_first=True,dropout=0,bidirectional=False)

    #初始化状态，否则torch默认全为0
    h_0 = torch.rand(num_layer,batch_size,hidden_size)
    c_0 = torch.rand(num_layer,batch_size,hidden_size)

    output,(h_n,c_n) = lstm(embed,(h_0,c_0))

    #非双向LSTM, num_direct=1
    #output [batch_size,            seq_len,    num_direc*hidden_size]
    #h_n    [num_direc*num_layer,   batch_size, hidden_size]
    #c_n    [num_direc*num_layer,   batch_size, hidden_size]

    print(output.shape,h_n.shape,c_n.shape)

    a = output[:,-1,:]
    b = h_n[1,:,:]
    c = h_n[0,:,:]
    print(a.shape)
    print(b.shape)
    print((a==b).sum())
    print(a==c)


#双向lstm
def double_lstm():
    batch_size = 10
    seq_len = 21
    embedding_dim = 30
    word_vocab = 100
    hidden_size = 18
    num_layer = 3
    num_direc = 2
    bidirectional = False
    if num_direc ==2:
        bidirectional = True

    #准备数据
    input = torch.randint(low=0,high=word_vocab,size=
                          [batch_size,seq_len])
    embedding = torch.nn.Embedding(num_embeddings=word_vocab,embedding_dim=embedding_dim)
    embed = embedding(input)

    #实例化双向LSTM
    lstm = torch.nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,
                         batch_first=True,dropout=0,bidirectional=bidirectional)

    #初始化
    h_0 = torch.rand(num_direc*num_layer,batch_size,hidden_size)
    c_0 = torch.rand(num_direc*num_layer,batch_size,hidden_size)

    #计算
    output,(h_n,c_n) = lstm(embed,(h_0,c_0))
    #output：seq_len 对应时间步，每步的输出实际上就是<最顶层>的隐状态，正向第一个拼接反向最后一个，正向第二个拼接反向倒数第二个,...
    #h_n：对于每一lstm层，正向计算到最后会得到一个隐状态，反向计算到最后也会得到一个隐状态，但它们并不像output那样拼接，
    #     而是按照先正向后反向进行堆叠，因此共有num_direc*num_layer层隐状态；此外，output保存了<最外层><每个时间步>的隐状态，而
    #     h_n保存的是<每层><正反向最后一个时间步>的隐状态(注意，是正向的最后一个和反向的最后一个，在output中也不是拼接到一起的)。
    #c_n: 细胞状态，类似于h_n


    print('size of output:',output.shape) #[batch_size,  seq_len,  num_direc*hidden_size]
    print('size of h_n:',h_n.shape) #[num_direc*num_layer,  batch_size,  hidden_size]
    print('size of c_n:',c_n.shape) #[num_direc*num_layer,  batch_size,  hidden_size]

    a1 = h_n[num_direc*num_layer-2,:,:] #取<最顶层>lstm正向的最后一个隐状态 [batch_size,hidden_size]
    a2 = h_n[num_direc*num_layer-1,:,:] #取<最顶层>lstm反向的最后一个隐状态 [batch_size,hidden_size]

    b1 = output[:,seq_len-1,0:hidden_size] # a1
    b2 = output[:,0,hidden_size:num_direc*hidden_size] # a2

    print(a1==b1)
    print(a2==b2)


if __name__ == '__main__':
    # single_lstm()
    double_lstm()
