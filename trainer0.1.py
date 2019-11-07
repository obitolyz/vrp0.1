import torch
import os
import Data_Generator
import torch.optim as optim
from PtrNet import NeuralCombOptRL
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np

np.set_printoptions(suppress=True, threshold=np.inf)


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser(description='vrptw with RL')
parser.add_argument('--dist_coef', default=1.0, type=float)
parser.add_argument('--over_cap_coef', default=0.1, type=float)
parser.add_argument('--over_time_coef', default=0.1, type=float)  ##
parser.add_argument('--cuda_device_id', default=0, type=int)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--num_node', default=21, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--train_size', default=12800, type=int)
parser.add_argument('--actor_net_lr', default=1e-3, type=float)
parser.add_argument('--critic_net_lr', default=1e-3, type=float)
parser.add_argument('--random_seed', default=111, type=int)
parser.add_argument('--use_cuda', default=True, type=str2bool)
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--disable_tensorboard', default=False, type=str2bool)

opt = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device_id
# torch.cuda.set_device(int(opt.cuda_device))
device = torch.device("cuda:{}".format(opt.cuda_device_id) if (torch.cuda.is_available() and opt.use_cuda) else "cpu")

# parameters
batch_size = opt.batch_size
train_size = opt.train_size
seq_len = opt.num_node  # service_num and depot num
n_epochs = opt.num_epoch
random_seed = opt.random_seed
actor_net_lr = opt.actor_net_lr
critic_net_lr = opt.critic_net_lr
load_path = opt.load_path  # model path
disable_tensorboard = opt.disable_tensorboard
dist_coef = opt.dist_coef
over_cap_coef = opt.over_cap_coef
over_time_coef = opt.over_time_coef

val_size = 1000
input_dim = 2
embedding_dim = 128
hidden_dim = 128
vehicle_init_capacity = 30
p_dim = 128  # as same to embedding_dim
R = 4
n_process_blocks = 3
n_glimpses = 1
use_tanh = True
C = 10  # tanh exploration
is_train = True
beam_size = 1  # if set B=1 then the technique is same as greedy search
output_dir = 'vrp_model/dist_{}_over_cap_{}_over_time_{}'.format(opt.dist_coef, opt.over_cap_coef, opt.over_time_coef)
log_dir = 'runs/dist_{}_over_cap_{}_over_time_{}'.format(opt.dist_coef, opt.over_cap_coef, opt.over_time_coef)

if not disable_tensorboard:
    writer = SummaryWriter(log_dir)

training_dataset = Data_Generator.VRPDataset(node_num=seq_len, num_samples=train_size)
# val_dataset = Data_Generator.VRPDataset(node_num=seq_len, num_samples=val_size)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# instantiate the Neural Combinatorial Opt with RL module
model = NeuralCombOptRL(embedding_dim,
                        hidden_dim,
                        seq_len,
                        n_glimpses,
                        n_process_blocks,
                        C,
                        use_tanh,
                        beam_size,
                        is_train,
                        device,
                        vehicle_init_capacity,
                        dist_coef,
                        over_cap_coef,
                        over_time_coef,
                        p_dim,
                        R)

# Load the model parameters from a saved state
if load_path != '':
    print('[*] Loading model from {}'.format(load_path))
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), load_path)))  # load parameters
    model.actor_net.decoder.seq_len = seq_len
    model.is_train = is_train

save_dir = os.path.join(os.getcwd(), output_dir)

try:
    os.makedirs(save_dir)
except:
    pass

critic_mse = torch.nn.MSELoss()
critic_optim = optim.Adam(model.critic_net.parameters(), lr=critic_net_lr)
actor_optim = optim.Adam(model.actor_net.parameters(), lr=actor_net_lr)

model = model.to(device)
critic_mse = critic_mse.to(device)

step = 0
val_step = 0
log_step = 5
epoch = 1000


def train_one_epoch(i):
    global step
    # put in train mode!
    model.train()

    # sample_batch is [batch_size x sourceL x input_dim]
    for batch_id, sample_batch in enumerate(tqdm(training_dataloader, disable=False)):
        sample_batch = sample_batch.to(device)
        R, b, probs, actions_idxs, dist_pc_pt = model(sample_batch)

        advantage = R - b  # means L(π|s) - b(s)

        # compute the sum of the log probs for each tour in the batch
        # probs: [2(seq_len-1)+1 x batch_size], logprobs: [batch_size]
        logprobs = sum([torch.log(prob) for prob in probs])
        # clamp any -inf's to 0 to throw away this tour
        # logprobs[(logprobs < -1000).detach()] = 0.  # means log p_(\theta)(π|s)
        logprobs = torch.clamp(logprobs, min=-10000)

        # multiply each time step by the advanrate
        reinforce = advantage * logprobs
        actor_loss = reinforce.mean()

        # actor net processing
        actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        # clip gradient norms
        torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(), max_norm=2.0, norm_type=2)
        actor_optim.step()

        # critic net processing
        R = R.detach()
        critic_loss = critic_mse(b, R)
        critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(), max_norm=2.0, norm_type=2)
        critic_optim.step()

        step += 1

        if not disable_tensorboard:
            writer.add_scalar('avg_reward', R.mean().item(), step)
            writer.add_scalar('actor_loss', actor_loss.item(), step)
            writer.add_scalar('critic_loss', critic_loss.item(), step)

        if step % log_step == 0:
            print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(i, batch_id, R.mean().item()))
            print('dist_pc_pt:{}, avg_dist_pc_pt:{}'.format(dist_pc_pt[-1], torch.mean(dist_pc_pt, dim=0)))
            print('solution:', actions_idxs.cpu().detach().numpy()[-1])
            # probs = torch.cat(probs, dim=0).view(-1, batch_size).permute(1, 0).cpu().detach().numpy()
            # print('Prob:', probs)

    return torch.mean(dist_pc_pt, dim=0)


# def validation():
#     global val_step
#     model.actor_net.decoder.decode_type = 'beam_search'
#     print('\n~Validating~\n')
#
#     example_input = []
#     example_output = []
#     avg_reward = []
#
#     # put in test mode!
#     model.eval()
#
#     for batch_id, val_batch in enumerate(tqdm(validation_dataloader, disable=False)):
#         val_batch = val_batch.to(device)
#
#         R, v, probs, actions, action_idxs = model(val_batch)
#
#         avg_reward.append(R[0].item())
#         val_step += 1
#
#         if not disable_tensorboard:
#             writer.add_scalar('val_avg_reward', R.item(), int(val_step))
#
#         if val_step % log_step == 0:
#             print('val_avg_reward:', R.item())
#
#             if plot_att:
#                 probs = torch.cat(probs, 0)
#                 plot_attention(example_input, example_output, probs.cpu().numpy())
#     print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
#     print('Validation overall reward var: {}'.format(np.var(avg_reward)))


def train_model():
    for i in range(epoch):
        if is_train:
            res = train_one_epoch(i)
        # Use beam search decoding for validation
        # validation()

        if is_train:
            model.actor_net.decoder.decode_type = 'stochastic'
            print('Saving model...epoch-{}.pt'.format(i))
            torch.save(model.state_dict(),
                       os.path.join(save_dir, '{}-D-{:.6f}-pc-{:.6f}-pt-{:.6f}.pt'.format(i, res[0], res[1], res[2])))


if __name__ == '__main__':
    train_model()
