import torch
import torch.nn as nn
import torch.nn.functional as F


class CharEmbedding(nn.Module):
    """
    CNN Based Character embedding
    """
    def __init__(self, d):
        super(CharEmbedding, self).__init__()
        self.d = d

    def forward(self, words):
        batch_size, length = words.size()
        return torch.randn(batch_size, length, self.d)

class BIDAF_Model(nn.Module):
    def __init__(self, vocab_size, d=100, dropout=0.2):
        super(BIDAF_Model, self).__init__()
        self.d = d
        self.char_embedding = CharEmbedding(d)
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size,
                                           embedding_dim=d)
        self.highway_layer =  nn.Linear(2*d,d) # TODO!
        self.lstm_H = nn.LSTM(input_size=d, hidden_size=d, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_U = nn.LSTM(input_size=d, hidden_size=d, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_M = nn.LSTM(input_size=8*d, hidden_size=d, num_layers=1,
                              batch_first=True, bidirectional=True)
        self.lstm_M2 = nn.LSTM(input_size=2*d, hidden_size=d, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.Ws = nn.Linear(6*d, 1, bias=False)
        self.Wp1 = nn.Linear(10*d, 1 , bias=False)
        self.Wp2 = nn.Linear(10*d, 1 , bias=False)
        pass

    def forward(self, contexts, querys):
        batch_size, T = contexts.size()
        _, J = querys.size()

        # Character Embed Layer
        context_char_embeds = self.char_embedding(contexts)
        print("context_char_embeds.size()=",context_char_embeds.size())
        # [batch, T, char embed size]
        query_char_embeds = self.char_embedding(querys)
        print("query_char_embeds.size()=",query_char_embeds.size())
        # [batch, J, char embed size]

        # Word Embed Layer
        context_word_embeds = self.word_embedding(contexts)
        print("context_word_embeds.size()=",context_word_embeds.size())
        # [batch, T, word embed size]
        query_word_embeds = self.word_embedding(querys)
        print("query_word_embeds.size()=",query_word_embeds.size())
        # [batch, J, word embed size]

        # Contextual Embed Layer
        X = torch.cat((context_word_embeds, context_char_embeds) ,dim=-1)
        X = self.highway_layer(X)
        print("X.size()=",X.size())
        # [N, T, d]
        Q = torch.cat((query_word_embeds, query_char_embeds) ,dim=-1)
        Q = self.highway_layer(Q)
        print("Q.size()=",Q.size())
        # [N, J, d]
        H, _ = self.lstm_H(X)
        print("H.size()=",H.size())
        # [N, T, 2d]
        U, _ = self.lstm_U(Q)
        print("U.size()=",U.size())
        # [N, J, 2d]

        # 4. Attention Flow Layer
        # similarity matrix
        shape = (batch_size, T, J, 2*self.d)
        H_expanded = H.unsqueeze(2)
        print("H_expanded.size()=",H_expanded.size())
        # [N, T, 1, 2d]
        H_expanded = H_expanded.expand(shape)
        print("H_expanded.size()=",H_expanded.size())
        # [N, T, J, 2d]
        U_expanded = U.unsqueeze(1)
        print("U_expanded.size()=",U_expanded.size())
        # [N, 1, J, 2d]
        U_expanded = U_expanded.expand(shape)
        print("U_expanded.size()=",U_expanded.size())
        # [N, T, J, 2d]
        HU = torch.mul(H_expanded, U_expanded)
        print("HU.size()=",HU.size())
        # [N, T, J, 2d]
        S = torch.cat((H_expanded, U_expanded, HU), dim=-1)
        print("S.size()=",S.size())
        # [N, T, J, 6d]
        S = self.Ws(S).view(batch_size, T, J)
        print("S.size()=",S.size())
        # [N, T, J]

        # Context-to-query Attention
        A = F.softmax(S, dim=2)
        print("A.size()=",A.size())
        # [N, T, J]
        U_tilde = torch.bmm(A,U)
        print("U_tilde.size()=",U_tilde.size())
        # [N,T,J] x [N,J,2d] -> [N, T, 2d]

        # Query-to-context Attention
        B = F.softmax(torch.max(S,dim=-1).values, dim=-1)
        print("B.size()=",B.size())
        # [N, T]
        h_tilde = torch.bmm(B.unsqueeze(1),H)
        print("h_tilde.size()=",h_tilde.size())
        # [N,1,T] x [N,T,2d] = [N,1,2d]
        
        shape = (batch_size, T, 2*self.d)
        H_tilde = h_tilde.expand(shape)
        print("H_tilde.size()=",H_tilde.size())
        # [N,T,2d]

        HU_tilde = H * U_tilde
        print("HU_tilde.size()=",HU_tilde.size())
        # [N,T,2d] * [N,T,2d] = [N,T,2d] (elementwise multiplication) 
        HH_tilde = H * H_tilde
        print("HH_tilde.size()=",HH_tilde.size())
        # [N,T,2d] * [N,T,2d] = [N,T,2d] (elementwise multiplication) 
        G = torch.cat((H, U_tilde, HU_tilde, HH_tilde) ,dim=-1)
        print("G.size()=",G.size())
        # [N,T,8d]

        # 5. Modeling Layer
        M, _ = self.lstm_M(G)
        print("M.size()=",M.size())
        # [N,T,2d]
        M2, _ = self.lstm_M2(M)
        print("M2.size()=",M2.size())
        # [N,T,2d]

        # 6. Output Layer
        P1 = F.softmax(self.Wp1(torch.cat((G, M), dim=-1)).squeeze(-1) , dim=1)
        print("P1.size()=",P1.size())
        # [N,T,1]
        P2 = F.softmax(self.Wp2(torch.cat((G, M2), dim=-1)).squeeze(-1) , dim=1)
        print("P2.size()=",P2.size())
        # [N,T,1]

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

def main():
    model = BIDAF_Model()
    contexts = torch.tensor([[1,2,3,4,5],[1,2,3,4,5]])
    querys = torch.tensor([[1,2,3,4],[1,2,3,4]])
    P1, P2 = model(contexts,querys)
    print("P1={}, P2={}".format(P1,P2))

if __name__ == "__main__":
    main()