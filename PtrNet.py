import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# torch.cuda.set_device(5)


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence to a hidden vector
    """

    def __init__(self, input_dim, hidden_dim, device):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.device = device
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        # hidden: (h0, c0)
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Parameter(torch.zeros(hidden_dim), requires_grad=False)
        enc_init_hx = enc_init_hx.to(self.device)

        # enc_init_hx.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Parameter(torch.zeros(hidden_dim), requires_grad=False)
        enc_init_cx = enc_init_cx.to(self.device)

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return enc_init_hx, enc_init_cx


# class Struct2Vec(nn.Module):
#     def __init__(self, node_num, device, p_dim, R):
#         super(Struct2Vec, self).__init__()
#         self.device = device
#         self.node_num = node_num
#         self.p_dim = p_dim
#         self.R = R
#         self.relu = torch.relu
#         self.theta_1 = nn.Linear(self.p_dim, self.p_dim, bias=False)  # mu
#         self.theta_2 = nn.Linear(self.p_dim, self.p_dim, bias=False)  # ll-w
#         self.theta_3 = nn.Linear(1, self.p_dim, bias=False)  # l-w
#
#         self.theta_4 = nn.Linear(5, self.p_dim, bias=False)  # service node
#         self.theta_5 = nn.Linear(2, self.p_dim, bias=False)  # depot node
#
#     def forward(self, inputs_origin):
#         """
#         :param inputs_origin: [sourceL x batch_size x input_dim], where input_dim: 5
#         :return: [sourceL x batch_size x embedded_dim]
#         """
#         inputs = inputs_origin.clone()
#         inputs[:, :, 2] = inputs[:, :, 2] / 10
#         inputs[:, :, 3] = inputs[:, :, 3] / 7
#         inputs[:, :, 4] = inputs[:, :, 4] / 7
#         batch_size = inputs.size(1)
#         N = self.node_num
#         mu = torch.zeros(N, batch_size, self.p_dim)
#         mu_null = torch.zeros(N, batch_size, self.p_dim)
#         mu = mu.to(self.device)
#             mu_null = mu_null.cuda()
#         for _ in range(self.R):
#             for i in range(N):
#                 item_1 = self.theta_1(torch.sum(mu, dim=0) - mu[i])
#                 item_2 = self.theta_2(sum(
#                     [self.relu(self.theta_3(torch.norm(inputs[i][:, :2] - inputs[j][:, :2], dim=1, keepdim=True))) for
#                      j in range(N)]))
#                 item_3 = self.theta_5(inputs[i][:, :2]) if i == 0 else self.theta_4(inputs[i])
#                 mu_null[i] = self.relu(item_1 + item_2 + item_3)
#             mu = mu_null.clone()
#
#         return mu

class Struct2Vec(nn.Module):
    def __init__(self, node_num, device, p_dim, R):
        super(Struct2Vec, self).__init__()
        self.device = device
        self.node_num = node_num

        self.theta_service = nn.Linear(5, p_dim)  # for service node
        self.theta_depot = nn.Linear(2, p_dim)  # for depot node

    def forward(self, inputs):
        """
        :param inputs: [batch_size x sourceL x input_dim], where input_dim: 5
        :return:
        """
        inputs[:, :, 2] = inputs[:, :, 2] / 10
        inputs[:, :, 3] = inputs[:, :, 3] / 7
        inputs[:, :, 4] = inputs[:, :, 4] / 7

        inputs = inputs.permute(1, 0, 2)
        depot = self.theta_depot(inputs[0, :, :2].unsqueeze(0))
        service = self.theta_service(inputs[1:, :, :])

        embedded_inputs = torch.cat([depot, service], dim=0)

        return embedded_inputs


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh, C, device):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(dim)
        v = v.to(device)
        self.v = nn.Parameter(v, requires_grad=True)
        self.v.data.uniform_(-1. / math.sqrt(dim), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current time step. [batch_size x hidden_dim]
            ref: the set of hidden states from the encoder.
                [sourceL x batch_size x hidden_dim]
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # [batch_size x hidden_dim x 1]
        e = self.project_ref(ref)  # [batch_size x hidden_dim x sourceL]
        # expand the query by sourceL
        # [batch x dim x sourceL]
        expanded_q = q.repeat(1, 1, e.size(2))
        # [batch x 1 x hidden_dim]
        v_view = self.v.unsqueeze(0).expand(expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL] = [batch_size x 1 x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        # if self.use_tanh:
        #     logits = self.C * self.tanh(u)
        # else:
        #     logits = u
        logits = u
        return e, logits


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 vehicle_init_capacity,
                 tanh_exploration,
                 use_tanh,
                 decode_type,
                 n_glimpses,
                 beam_size,
                 device):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.decode_type = decode_type
        self.beam_size = beam_size
        self.device = device

        self.vehicle_init_capacity = vehicle_init_capacity

        # self.input_weights = nn.Linear(embedding_dim + 2, 4 * hidden_dim)  # modify the decoder_input dimension
        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration, device=device)
        self.glimpse = Attention(hidden_dim, use_tanh=False, C=tanh_exploration, device=device)
        self.sm = nn.Softmax(dim=1)

    def apply_mask_to_logits(self, logits, mask, prev_idxs):
        if mask is None:
            if torch.__version__ == '1.3.0':
                mask = torch.zeros(logits.size()).bool()
            else:
                mask = torch.zeros(logits.size()).byte()
            mask = mask.to(self.device)

        maskk = mask.clone()

        # to prevent them from being reselected.
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[list(range(logits.size(0))), prev_idxs] = 1  # awesome!
            # maskk[torch.nonzero(prev_idxs).squeeze(1), 0] = 0  # filter
            # logits[maskk] = -np.inf
            maskk[:, 0] = 0
            for i in range(maskk.size(0)):
                if prev_idxs[i] == 0 and 0 in maskk[i, 1:]:
                    maskk[i, 0] = 1
            logits.masked_fill_(maskk, -np.inf)  # nondifferentiable?

        return logits, maskk

    def forward(self, decoder_input, before_embedded_inputs, embedded_inputs, hidden, context):
        """
        Args:
            :param context: encoder outputs, [sourceL x batch_size x hidden_dim]
            :param hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            :param decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            :param embedded_inputs: [sourceL x batch_size x embedding_dim]
            :param before_embedded_inputs:
        """

        def recurrence(x, hidden, logit_mask, prev_idxs):

            hx, cx = hidden  # batch_size x hidden_dim
            # gates: [batch_size x (hidden_dim x 4)]
            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            # for _ in range(self.n_glimpses):
            #     ref, logits = self.glimpse(g_l, context)
            #     logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask, prev_idxs)
            #     # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = [batch_size x h_dim x 1]
            #     g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(logits, logit_mask, prev_idxs)
            probs = self.sm(logits)

            return hy, cy, probs, logit_mask

        batch_size = context.size(1)
        outputs = []
        selections = []
        mask = None
        idxs = torch.LongTensor([0] * batch_size).to(self.device)  #
        # selections.append(idxs)  #
        # choose_i = torch.LongTensor([0]).to(device)
        # prob_0 = torch.zeros(batch_size, self.seq_len).to(device)
        # prob_0.index_fill_(1, choose_i, 1)
        # prob_0.requires_grad = True  ###
        # outputs.append(prob_0)

        # record (remaining capacity, current time, distance, penalty_capacity, penalty_time)
        rc_ct_d_pc_pt = torch.FloatTensor([self.vehicle_init_capacity, 0, 0, 0, 0]).repeat(batch_size, 1).detach_()
        rc_ct_d_pc_pt = rc_ct_d_pc_pt.to(self.device)

        if self.decode_type == 'stochastic':
            # at most twice (seq_len - 1), seq_len: service_num + depot_num
            for _ in range((self.seq_len - 1) * 2):
                decoder_input = decoder_input.to(self.device)
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs, rc_ct_d_pc_pt = self.decode_stochastic(probs, before_embedded_inputs,
                                                                            embedded_inputs, idxs, rc_ct_d_pc_pt)

                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)

            return (outputs, selections), hidden, rc_ct_d_pc_pt[:, -3:]

        elif self.decode_type == 'greedy':
            # embedded_inputs: [sourceL x batch_size x embedding_dim]
            # decoder_input: [batch_size x embedding_dim]
            # hidden: [batch_size x hidden_dim]
            # context: [sourceL x batch_size x hidden_dim]
            pass

    def decode_stochastic(self, probs, before_embedded_inputs, embedded_inputs, prev_idxs, rc_ct_d_pc_pt):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        idxs = probs.multinomial(1).squeeze(1)

        # remaining capacity, current time
        # if vehicle returns to depot 0, set remaining capacity to vehicle_init_capacity, current time to 0

        # nonzero
        nonzero_idxs = torch.nonzero(idxs).squeeze(1)
        # zero
        zero_idxs = torch.from_numpy(np.where(idxs.cpu() == 0)[0])

        prev_x_y = before_embedded_inputs[prev_idxs, list(range(batch_size)), :2]
        cur_x_y = before_embedded_inputs[idxs, list(range(batch_size)), :2]
        distance = torch.norm(prev_x_y - cur_x_y, dim=1)

        # remaining capacity
        required_capacity = before_embedded_inputs[idxs, list(range(batch_size)), 2]
        t_1 = before_embedded_inputs[idxs, list(range(batch_size)), 3]
        t_2 = before_embedded_inputs[idxs, list(range(batch_size)), 4]

        # current time
        for i in range(batch_size):
            if i in nonzero_idxs:
                cur_t = rc_ct_d_pc_pt[i][1] + distance[i]
                if cur_t < t_1[i]:
                    rc_ct_d_pc_pt[i][1] = t_1[i]
                else:
                    rc_ct_d_pc_pt[i][1] = cur_t

                rc_ct_d_pc_pt[i][0] = rc_ct_d_pc_pt[i][0] - required_capacity[i]
                rc_ct_d_pc_pt[i][-1] = rc_ct_d_pc_pt[i][-1] + max(rc_ct_d_pc_pt[i][1] - t_2[i], 0)  # penalty time
                rc_ct_d_pc_pt[i][-2] = rc_ct_d_pc_pt[i][-2] + max(-rc_ct_d_pc_pt[i][0], 0)  # penalty capacity

        rc_ct_d_pc_pt[:, 2] = rc_ct_d_pc_pt[:, 2] + distance  # cumulative distance
        reset_rc_ct = torch.FloatTensor([self.vehicle_init_capacity, 0])
        reset_rc_ct = reset_rc_ct.to(self.device)
        rc_ct_d_pc_pt[zero_idxs, :2] = reset_rc_ct  # reset capacity and time

        # sels = torch.zeros(batch_size, self.embedding_dim + 2)
        # sels[:, :-2] = embedded_inputs[idxs, list(range(batch_size)), :]  # [batch_size x embedding_size]
        # sels[:, -2:] = rc_ct_d_pc_pt[:, :2].clone()  # clone part of rc_ct_pc_pt
        sels = embedded_inputs[idxs, list(range(batch_size)), :]

        return sels, idxs, rc_ct_d_pc_pt


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq model
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 beam_size,
                 device,
                 vehicle_init_capacity,
                 p_dim,
                 R):
        super(PointerNetwork, self).__init__()

        self.device = device
        self.vehicle_init_capacity = vehicle_init_capacity
        self.embedding_dim = embedding_dim

        self.s2v = Struct2Vec(seq_len, device, p_dim, R)

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            device)

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            seq_len,
            vehicle_init_capacity,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            decode_type='stochastic',
            n_glimpses=n_glimpses,
            beam_size=beam_size,
            device=device)

        # Trainable initial hidden states
        dec_in_0 = torch.FloatTensor(embedding_dim)
        dec_in_0 = dec_in_0.to(device)

        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-1. / math.sqrt(embedding_dim), 1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args:
            inputs: [batch_size x sourceL x input_dim]
        """
        # [sourceL x batch_size x embedded_dim]
        embedded_inputs = self.s2v(inputs.clone())

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(
            0)  # [1 x batch_size x hidden_dim]
        encoder_cx = encoder_cx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        # context: [seq_len x batch_size x hidden_dim], enc_h_t: [1 x batch_size x hidden_dim]
        context, (enc_h_t, enc_c_t) = self.encoder(embedded_inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # decoder_input: [batch_size x embedding_dim]
        # decoder_input = torch.zeros(embedded_inputs.size(1), self.embedding_dim + 2)  # remaining capacity, current time
        # decoder_input[:, -2] = self.vehicle_init_capacity
        # decoder_input[:, :-2] = embedded_inputs[0].clone()
        # decoder_input = decoder_input.detach()
        decoder_input = embedded_inputs[0]

        (pointer_probs, input_idxs), dec_hidden_t, dist_pc_pt = self.decoder(decoder_input,
                                                                             inputs.permute(1, 0, 2),
                                                                             embedded_inputs,
                                                                             dec_init_state,
                                                                             context)

        return pointer_probs, input_idxs, dist_pc_pt


class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_blocks,
                 tanh_exploration,
                 use_tanh,
                 device,
                 seq_len,
                 p_dim,
                 R):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_blocks = n_process_blocks

        self.s2v = Struct2Vec(seq_len, device, p_dim, R)

        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               device)

        self.process_block = Attention(hidden_dim,
                                       use_tanh=use_tanh,
                                       C=tanh_exploration,
                                       device=device)
        self.sm = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # baseline prediction, a single scalar
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x sourceL x embedded_dim]
        """
        # [sourceL x batch_size x embedded_dim]
        embedded_inputs = self.s2v(inputs.clone())

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state  # [hidden_dim]
        encoder_hx = encoder_hx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(
            0)  # [1 x batch_size x hidden_dim]
        encoder_cx = encoder_cx.unsqueeze(0).repeat(embedded_inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(embedded_inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]  # [batch_size x hidden_dim]
        for _ in range(self.n_process_blocks):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class NeuralCombOptRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and CriticNetwork (critic).
    It requires an application-specific reward function
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 n_glimpses,
                 n_process_blocks,
                 tanh_exploration,  # C
                 use_tanh,
                 beam_size,
                 is_train,
                 device,
                 vehicle_init_capacity,
                 dist_coef,
                 over_cap_coef,
                 over_time_coef,
                 p_dim,
                 R):
        super(NeuralCombOptRL, self).__init__()
        self.is_train = is_train
        self.device = device
        self.dist_coef = dist_coef
        self.over_cap_coef = over_cap_coef
        self.over_time_coef = over_time_coef

        self.actor_net = PointerNetwork(
            embedding_dim,
            hidden_dim,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            beam_size,
            device,
            vehicle_init_capacity,
            p_dim,
            R)

        # utilize critic network
        self.critic_net = CriticNetwork(
            embedding_dim,
            hidden_dim,
            n_process_blocks,
            tanh_exploration,
            False,
            device,
            seq_len,
            p_dim,
            R)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x sourceL x input_dim]
        """
        batch_size = inputs.size(0)

        # query the actor net for the input indices
        # making up the output, and the pointer attn
        probs_, action_idxs, dist_pc_pt = self.actor_net(inputs)
        # probs_: [seq_len x batch_size x seq_len], action_idxs: [seq_len x batch_size]

        if self.is_train:
            # probs_ is a list of len sourceL of [batch_size x sourceL]
            # probs: [sourceL x batch_size]
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                probs.append(prob[list(range(batch_size)), action_id])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            probs = probs_

        cff = torch.FloatTensor([self.dist_coef, self.over_cap_coef, self.over_time_coef])
        cff = cff.to(self.device)

        R = torch.mm(dist_pc_pt, cff.view(-1, 1)).squeeze(1)  # calc reward
        # get the critic value fn estimates for the baseline
        # [batch_size]
        b = self.critic_net(inputs).squeeze(1)

        action_idxs = torch.cat(action_idxs, 0).view(-1, batch_size).transpose(1, 0)

        return R, b, probs, action_idxs, dist_pc_pt
