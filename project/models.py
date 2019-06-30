import torch
from torch import nn
from torch.optim import Adam
import numpy as np


class Net(nn.Module):
    def __init__(self, args, lossweight):
        super(Net, self).__init__()
        self.lr = args.lr
        self.model = args.model
        self.alpha = args.alpha
        self.use_cpu = args.cpu

        self.n_class = args.num_class
        self.embedding = nn.Embedding(num_embeddings=args.num_class, embedding_dim=args.emb_dim)
        self.emb_drop = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTM(input_size=args.emb_dim + 1,
                            hidden_size=args.hid_dim,
                            batch_first=True,
                            bidirectional=False)
        self.mlp = nn.Linear(in_features=args.hid_dim, out_features=args.mlp_dim)
        self.mlp_drop = nn.Dropout(p=args.dropout)
        self.event_output = nn.Linear(in_features=args.mlp_dim, out_features=args.num_class)
        self.time_output = nn.Linear(in_features=args.mlp_dim, out_features=1)
        self.set_criterion(lossweight)

    def set_optimizer(self, total_step):
        self.optimizer = Adam(self.parameters(), lr=self.lr)

    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        if self.model == 'rmtpp':
            if not self.use_cpu:
                self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
                self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            else:
                self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
                self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
            self.time_criterion = self.RMTPPLoss
        else:
            self.time_criterion = nn.MSELoss()

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w)
        return -1 * loss

    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1)
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_logits = self.event_output(mlp_output)
        time_logits = self.time_output(mlp_output)
        return time_logits, event_logits

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            if self.use_cpu:
                tensors[i] = tensors[i].contiguous()
            else:
                tensors[i] = tensors[i].cuda().contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        time_logits, event_logits = self.forward(time_input, event_input)
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.alpha * loss1 + loss2
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred
