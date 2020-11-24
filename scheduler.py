
class PolyScheduler(object):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warmup_ratio=1e-2, warmup_lr_ratio=1e-3):
        self.optimizer        = optimizer
        self.base_lr          = base_lr
        self.power            = power
        self.warmup_lr_ratio  = warmup_lr_ratio
        self.warmup_ratio     = warmup_ratio
        self.max_iters        = max_iters

        self.warmup_end_iter  = int(max_iters * warmup_ratio)
        self.warmup_lr        = base_lr * warmup_lr_ratio
        self.c_iters          = 0

    def get_lr(self, nstep):
        if nstep < self.warmup_end_iter:
            ratio = nstep / self.warmup_end_iter
            return self.warmup_lr + (ratio ** self.power) * (self.base_lr - self.warmup_lr)
        else:
            n_total_step = self.max_iters - self.warmup_end_iter
            ratio = (nstep - self.warmup_end_iter) / n_total_step
            return self.base_lr * ((1-ratio) ** self.power)

    def step(self):
        lr = self.get_lr(self.c_iters)
        self.c_iters += 1
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr