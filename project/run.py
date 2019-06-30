from argparse import ArgumentParser
from torch.utils.data import DataLoader

import numpy as np

import json
import pickle

from tqdm import tqdm
import utils
from models import Net


def getParser():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="pp")
    parser.add_argument("--model", type=str, default="erpp", help="erpp, rmtpp")
    parser.add_argument("--seq_len", type=int, default=10)

    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num_class", type=int, default=7)
    parser.add_argument("--echo_every", type=int, default=350)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--run_epochs", type=int, default=30)


    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = getParser()

    train_set = utils.MyDataset(args, subset='train')
    test_set = utils.MyDataset(args, subset='test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=utils.MyDataset.to_features)
    test_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=utils.MyDataset.to_features)

    weight = train_set.init_weight()
    model = Net(args, lossweight=weight)
    model.set_optimizer(total_step=len(train_loader) * args.run_epochs)
    if not args.cpu:
        model.cuda()

    MAE = []
    presicion = []
    recall = []
    f1 = []
    result = {'MAE': MAE, 'presicion': presicion, 'recall': recall, 'f1': f1}

    print("test for saving result at: ", "result/"+args.name+".json")
    pickle.dump(result, open("result/"+args.name+".json", "wb"))

    for i in range(args.run_epochs):
        model.train()
        loss = []
        for j, data in enumerate(tqdm(train_loader)):
            tmp = model.train_batch(data)
            loss.append(tmp)
            if (j+1) % args.echo_every == 0:
                print("train loss: {}".format(np.mean(loss[-args.echo_every:], axis=0)))
        utils.evaluation(i, args, model, test_loader, result)
        pickle.dump(result, open("result/"+args.name+".json", "wb"))
