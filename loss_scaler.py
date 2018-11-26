import torch

class LossScaler:

    def __init__(self, scale=1):
        self.cur_scale = scale

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        return False

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()

class DynamicLossScaler:

    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
#        return False
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        cpu_sum = float(x.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        return False

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        if overflow:
            #self.cur_scale /= self.scale_factor
            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.cur_scale *= self.scale_factor
#        self.cur_scale = 1
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()

##############################################################
# Example usage below here -- assuming it's in a separate file
##############################################################
if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    from dynamic_loss_scaler import DynamicLossScaler

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(torch.randn(N, D_in), requires_grad=False)
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    w1 = Variable(torch.randn(D_in, H), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out), requires_grad=True)
    parameters = [w1, w2]

    learning_rate = 1e-6
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_scaler = DynamicLossScaler()

    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum() * loss_scaler.loss_scale
        print('Iter {} loss scale: {}'.format(t, loss_scaler.loss_scale))
        print('Iter {} scaled loss: {}'.format(t, loss.data[0]))
        print('Iter {} unscaled loss: {}'.format(t, loss.data[0] / loss_scaler.loss_scale))

        # Run backprop
        optimizer.zero_grad()
        loss.backward()

        # Check for overflow
        has_overflow = DynamicLossScaler.has_overflow(parameters)

        # If no overflow, unscale grad and update as usual
        if not has_overflow:
            for param in parameters:
                param.grad.data.mul_(1. / loss_scaler.loss_scale)
            optimizer.step()
        # Otherwise, don't do anything -- ie, skip iteration
        else:
            print('OVERFLOW!')

        # Update loss scale for next iteration
        loss_scaler.update_scale(has_overflow)

