import torch
from torch.optim import Optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Bayes_SAM(Optimizer):
    """
    Implement Bayes-SAM
    """
    
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, rho=0.1, gamma=0.1, wdecay=0.0001, s_init=1., Ndata = 100):
        defaults = dict(lr = lr, beta1 = beta1, beta2=beta2, rho=rho, gamma=gamma, wdecay=wdecay, s_init = s_init, Ndata = Ndata)
        super().__init__(params, defaults)

        self.init_state()
    
    def init_state(self):
        state  = {}
        for param_group in self.param_groups:
            s_init = param_group['s_init']
            for p in param_group['params']:
                state[p] = {'s': s_init * torch.ones_like(p),
                            'gm': torch.zeros_like(p),
                            'mean': p.clone(),
                            'g_noise': torch.zeros_like(p),
                            'g_perturb': torch.zeros_like(p)}
        
        self.state = state
    
    
    @torch.no_grad()
    def reset_param(self):
        """
        set parameters to mean value
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.grad is None: continue
                p.data = self.state[p]['mean'].clone() # move to w
                
    @torch.no_grad()
    def reset_gradient_state(self):
        """
        reset gradients values to zero
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.grad is None: continue
                self.state[p]['g_noise'] = torch.zeros_like(p)
                self.state[p]['g_perturb'] = torch.zeros_like(p)
                
    @torch.no_grad()
    def update_g_noise(self, k):
        """
        we use running averaging to calculate ∇L(w+e)
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.grad is None: continue
                self.state[p]['g_noise'] = (k * self.state[p]['g_noise'] + p.grad) / (k + 1)
                
    @torch.no_grad()
    def update_g_perturb(self, k):
        """
        we use running averaging to calculate ∇L(w+ϵ)
        """
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.grad is None: continue
                self.state[p]['g_perturb'] = (k * self.state[p]['g_perturb'] + p.grad) / (k + 1)
    
    @torch.no_grad()
    def add_noise(self):
        """
        set the parameter from w to (w + e) where e ~ N(0, σ^2), σ^2 = 1 / (N·s)
        """
        for param_group in self.param_groups:
            Ndata = param_group['Ndata']
            for p in param_group['params']:
                if p.grad is None: continue
                noise = torch.randn(size=p.size(), device=device) / torch.sqrt(Ndata * self.state[p]['s']) # sample noise e
                p.data = self.state[p]['mean'].clone() # move to w
                p.data +=noise # move to w + e
    
        
    @torch.no_grad()
    def add_perturbation(self):
        """
        set the parameter from w to (w + ϵ) where ϵ = ρ * g_noise / s, g_noise=∇L(w+e)
        """
        for param_group in self.param_groups:
            rho = param_group['rho']
            for p in param_group['params']:
                if p.grad is None: continue
                perturbation = rho * self.state[p]['g_noise'] / self.state[p]['s']# calculate ϵ
                p.data = self.state[p]['mean'].clone() # move to w
                p.data += perturbation # move to w + ϵ
            
    def step(self, lr):
        """
        update mean parameter w
        """
        for param_group in self.param_groups:
            beta1 = param_group['beta1']
            beta2 = param_group['beta2']
            wdecay = param_group['wdecay']
            gamma = param_group['gamma']
            for p in param_group['params']:
                if p.grad is None: continue
                p.data = self.state[p]['mean'].clone() # move to w
                self.state[p]['gm'] = beta1 * self.state[p]['gm'] + (1 - beta1) * (self.state[p]['g_perturb'] + wdecay * p) # update gm
                self.state[p]['s'] = beta2 * self.state[p]['s'] + \
                            (1 - beta2) * (torch.sqrt(self.state[p]['s'] * self.state[p]['g_noise'] ** 2) + wdecay + gamma) # update s
                p.data -= lr * self.state[p]['gm'] / self.state[p]['s'] # update w
                self.state[p]['mean'] = p.data.clone() # update state value
