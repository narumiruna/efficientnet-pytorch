class GradualWarmup(object):

    def __init__(self, optimizer, total_iterations=400):
        self.optimizer = optimizer
        self.base_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        self.total_iterations = total_iterations
        self.last_iteration = 0

    def step(self):
        self.last_iteration += 1

        if self.last_iteration > self.total_iterations:
            return

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * self.last_iteration / self.total_iterations
