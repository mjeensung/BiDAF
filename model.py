import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
LOGGER = logging.getLogger()

class Highway_Network(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(Highway_Network, self).__init__()
        self.transform_gate = nn.Linear(input_dim,output_dim,bias=True)
        self.highway_gate = nn.Linear(input_dim,output_dim,bias=True)

    def forward(self,input):
        # print("input.size()=",input.size())
        # [N, sentence_len, input_dim]
        
        transform_gate = torch.sigmoid(self.transform_gate(input))
        # print("transform_gate.size()=",transform_gate.size())
        # [N, sentence_len, output_dim]
        carry_gate = 1 - transform_gate
        # print("carry_gate.size()=",carry_gate.size())
        # [N, sentence_len, output_dim]
        highway_gate = F.relu(self.highway_gate(input))
        # print("highway_gate.size()=",highway_gate.size())
        # [N, sentence_len, output_dim]

        return transform_gate * highway_gate + carry_gate * input

    def cuda(self):
        self.transform_gate = self.transform_gate.cuda()
        self.highway_gate = self.highway_gate.cuda()

        return self
    
class CharEmbedding(nn.Module):
    """
    CNN Based Character embedding
    """
    def __init__(self, char_size, char_dim, dropout):
        super(CharEmbedding, self).__init__()
        self.charembed = nn.Embedding(char_size, char_dim, padding_idx=0)

        self.width = 5
        self.outchannel = char_dim
        self.d = char_dim
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.outchannel, kernel_size=(self.width,self.d))
        self.dropout = nn.Dropout(dropout)
        self.highway_layer = Highway_Network(self.outchannel,self.outchannel)

    def forward(self, chars):
        """
        In : (N, sentence_len, word_len)
        Out: (N, sentence_len, d)
        """
        # print("chars=",chars)
        batch_size, sentence_len, word_len = chars.size()
        
        x = chars.view(-1, word_len) # append all batchs(sentences) in a row
        # print("x1.size()=",x.size())
        # [N* sentence_len, word_len]

        x = self.charembed(x)
        # print("x2.size()=",x.size())
        # [N* sentence_len, word_len, d]

        x = x.unsqueeze(1)
        # print("x3.size()=",x.size())
        # [N* sentence_len, inchannel, word_len, d]
        
        x = self.conv(x)
        # print("x4.size()=",x.size())
        # [N* sentence_len, outchannel, featuremap_size, 1]

        x = x.squeeze().permute(0,2,1)
        # print("x5.size()=",x.size())
        # [N* sentence_len, featuremap_size, outchannel]

        x = torch.tanh(x)
        x = self.dropout(x)
        # print("x6.size()=",x.size())
        # [N* sentence_len, featuremap_size, outchannel]

        # max pooling
        x = torch.max(x, dim=-2).values
        # print("x7.size()=",x.size())
        # [N* sentence_len, 100]

        # highway layer
        x = self.highway_layer(x)
        # print("x8.size()=",x.size())
        # [N* sentence_len, 100]

        x = x.view(batch_size, sentence_len, -1)
        # print("x9.size()=",x.size())
        # [N, sentence_len, 100]

        return x

    def cuda(self):
        self.charembed = self.charembed.cuda()
        self.conv = self.conv.cuda()
        self.highway_layer = self.highway_layer.cuda()

        return self

class BIDAF_Model(nn.Module):
    def __init__(self, char_size, vocab_size, char_dim, word_dim, dropout=0.2, use_cuda=True):
        super(BIDAF_Model, self).__init__()
        self.d = char_dim + word_dim
        self.char_embedding = CharEmbedding(char_size= char_size, char_dim=char_dim, dropout=dropout)
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=word_dim, padding_idx=0)
        self.highway_layer =  nn.Sequential(
            Highway_Network(self.d,self.d),
            Highway_Network(self.d,self.d))
        self.lstm_H = nn.LSTM(input_size=self.d, hidden_size=self.d, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_U = nn.LSTM(input_size=self.d, hidden_size=self.d, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_M = nn.LSTM(input_size=8*self.d, hidden_size=self.d, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_M2 = nn.LSTM(input_size=2*self.d, hidden_size=self.d, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.Ws = nn.Linear(6*self.d, 1, bias=False)
        self.Wp1 = nn.Linear(10*self.d, 1 , bias=False)
        self.Wp2 = nn.Linear(10*self.d, 1 , bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_cuda = use_cuda

        if use_cuda:
            self.char_embedding = self.char_embedding.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.highway_layer = self.highway_layer.cuda()
            self.lstm_H = self.lstm_H.cuda()
            self.lstm_U = self.lstm_U.cuda()
            self.lstm_M = self.lstm_M.cuda()
            self.lstm_M2 = self.lstm_M2.cuda()
            self.Ws = self.Ws.cuda()
            self.Wp1 = self.Wp1.cuda()
            self.Wp2 = self.Wp2.cuda()
            

    def forward(self, context_words, context_chars, query_words, query_chars):
        if self.use_cuda:
            context_words = context_words.cuda()
            context_chars = context_chars.cuda()
            query_words = query_words.cuda()
            query_chars = query_chars.cuda()
        batch_size, T = context_words.size()
        _, J = query_words.size()

        # Character Embed Layer
        context_char_embeds = self.char_embedding(context_chars)
        # print("context_char_embeds.size()=",context_char_embeds.size())
        # [batch, T, char embed size]
        query_char_embeds = self.char_embedding(query_chars)
        # print("query_char_embeds.size()=",query_char_embeds.size())
        # [batch, J, char embed size]

        # Word Embed Layer
        context_word_embeds = self.word_embedding(context_words)
        # print("context_word_embeds.size()=",context_word_embeds.size())
        # [batch, T, word embed size]
        query_word_embeds = self.word_embedding(query_words)
        # print("query_word_embeds.size()=",query_word_embeds.size())
        # [batch, J, word embed size]

        # Contextual Embed Layer
        X = torch.cat((context_word_embeds, context_char_embeds) ,dim=-1)
        # print("X.size()=",X.size())
        # [N, T, 2*d]
        X = self.highway_layer(X)
        # print("X.size()=",X.size())
        # [N, T, d]
        Q = torch.cat((query_word_embeds, query_char_embeds) ,dim=-1)
        Q = self.highway_layer(Q)
        # print("Q.size()=",Q.size())
        # [N, J, d]
        H, _ = self.lstm_H(X)
        H = self.dropout(H)
        # print("H.size()=",H.size())
        # [N, T, 2d]
        U, _ = self.lstm_U(Q)
        U = self.dropout(U)
        # print("U.size()=",U.size())
        # [N, J, 2d]

        # 4. Attention Flow Layer
        # similarity matrix
        shape = (batch_size, T, J, 2*self.d)
        H_expanded = H.unsqueeze(2)
        # print("H_expanded.size()=",H_expanded.size())
        # [N, T, 1, 2d]
        H_expanded = H_expanded.expand(shape)
        # print("H_expanded.size()=",H_expanded.size())
        # [N, T, J, 2d]
        U_expanded = U.unsqueeze(1)
        # print("U_expanded.size()=",U_expanded.size())
        # [N, 1, J, 2d]
        U_expanded = U_expanded.expand(shape)
        # print("U_expanded.size()=",U_expanded.size())
        # [N, T, J, 2d]
        HU = torch.mul(H_expanded, U_expanded)
        # print("HU.size()=",HU.size())
        # [N, T, J, 2d]
        S = torch.cat((H_expanded, U_expanded, HU), dim=-1)
        # print("S.size()=",S.size())
        # [N, T, J, 6d]
        S = self.Ws(S).view(batch_size, T, J)
        # print("S.size()=",S.size())
        # [N, T, J]

        # Context-to-query Attention
        A = F.softmax(S, dim=2)
        # print("A.size()=",A.size())
        # [N, T, J]
        U_tilde = torch.bmm(A,U)
        # print("U_tilde.size()=",U_tilde.size())
        # [N,T,J] x [N,J,2d] -> [N, T, 2d]

        # Query-to-context Attention
        B = F.softmax(torch.max(S,dim=-1).values, dim=-1)
        # print("B.size()=",B.size())
        # [N, T]
        h_tilde = torch.bmm(B.unsqueeze(1),H)
        # print("h_tilde.size()=",h_tilde.size())
        # [N,1,T] x [N,T,2d] = [N,1,2d]
        
        shape = (batch_size, T, 2*self.d)
        H_tilde = h_tilde.expand(shape)
        # print("H_tilde.size()=",H_tilde.size())
        # [N,T,2d]

        HU_tilde = H * U_tilde
        # print("HU_tilde.size()=",HU_tilde.size())
        # [N,T,2d] * [N,T,2d] = [N,T,2d] (elementwise multiplication) 
        HH_tilde = H * H_tilde
        # print("HH_tilde.size()=",HH_tilde.size())
        # [N,T,2d] * [N,T,2d] = [N,T,2d] (elementwise multiplication) 
        G = torch.cat((H, U_tilde, HU_tilde, HH_tilde) ,dim=-1)
        # print("G.size()=",G.size())
        # [N,T,8d]

        # 5. Modeling Layer
        M, _ = self.lstm_M(G)
        M = self.dropout(M)
        # print("M.size()=",M.size())
        # [N,T,2d]
        M2, _ = self.lstm_M2(M)
        M2 = self.dropout(M2)
        # print("M2.size()=",M2.size())
        # [N,T,2d]

        # 6. Output Layer
        P1 = self.Wp1(torch.cat((G, M), dim=-1)).squeeze(-1)
        P1 = self.dropout(P1)
        P1 = F.softmax(P1 , dim=1)
        # print("P1.size()=",P1.size())
        # [N,T]
        P2 = self.Wp2(torch.cat((G, M2), dim=-1)).squeeze(-1)
        P2 = self.dropout(P2)
        P2 = F.softmax(P2, dim=1)
        # print("P2.size()=",P2.size())
        # [N,T]

        return P1, P2
        # return 0, 0

    def get_loss(self, start_idx, end_idx, p1, p2):
        # print("start_idx=",start_idx)
        # print("p1=",p1)
        P1 = p1[:,start_idx]
        P2 = p2[:,end_idx]
        # print("P1=",P1)
        N = len(P1)
        
        return -(1/N)*torch.sum(torch.log(P1)+torch.log(P2))

    def save_checkpoint(self, state, checkpoint_dir, filename):    
        filename = checkpoint_dir + filename
        LOGGER.info('Save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        LOGGER.info('Load checkpoint %s' % filename)
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return self

def main():
    model = BIDAF_Model(char_size=20, vocab_size=100)
    model.eval()
    context_chars = torch.tensor([[[1,2,3,4,5,6,0],[1,2,3,4,5,6,0]]])
    query_chars = torch.tensor([[[1,2,3,4,5,6],[1,2,3,4,5,6]]])
    context_words = torch.tensor([[1,2]])
    query_words = torch.tensor([[1,2]])
    P1, P2 = model(context_chars=context_chars,query_chars=query_chars,context_words=context_words,query_words=query_words)
    print("P1={}, P2={}".format(P1,P2))

if __name__ == "__main__":
    main()