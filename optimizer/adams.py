import torch
from torch.optim.optimizer import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """
        Adam optimizer implementation.

        Args:
            params (iterable): Parameters to optimize.
            lr (float): Learning rate.
            betas (tuple): Coefficients for moving averages of gradient and its square.
            eps (float): Small term added to denominator for numerical stability.
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss: The loss value, if the closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            learning_rate = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
              
                state = self.state[param]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['first_moment'] = torch.zeros_like(param.data)
                    state['second_moment'] = torch.zeros_like(param.data)

                first_moment = state['first_moment']
                second_moment = state['second_moment']
                state['step'] += 1

                step_count = state['step']

                # Update biased first and second moment estimates
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * (grad * grad)

                # Bias correction
                corrected_first_moment = first_moment / (1 - beta1 ** step_count)
                corrected_second_moment = second_moment / (1 - beta2 ** step_count)

                # Compute parameter update
                update = corrected_first_moment / (corrected_second_moment.sqrt() + epsilon)
                param.data -= learning_rate * update

                # Save updated moment estimates
                state['first_moment'] = first_moment
                state['second_moment'] = second_moment

        return loss
