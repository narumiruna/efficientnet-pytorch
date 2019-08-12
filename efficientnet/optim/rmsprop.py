import torch
from torch import optim


class TFRMSprop(optim.Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=0, alpha=0, eps=1e-8, momentum=0):
        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps, momentum=momentum)
        super(TFRMSprop, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['squared_grad_buffer'] = torch.zeros_like(p)
                    state['momentum_buffer'] = torch.zeros_like(p)

                # weight decay
                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # rmsprop
                state['step'] += 1
                alpha = group['alpha']
                if alpha != 0:
                    buf = state['squared_grad_buffer']
                    buf.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                    corrected = buf.div(1 - alpha**state['step'])
                    d_p.div_(corrected.sqrt() + group['eps'])

                if group['momentum'] != 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(group['lr'], d_p)
                    p.data.add_(-buf)
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss
