import torch

class Config:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


class MPPI(torch.nn.Module):
    """
    Model Predictive Path Integral control
    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    """

    def __init__(
        self,
        Predmodel,
        Dynamics,
        Costs,
        Sampling,
        MPPI_config,
        device="cuda:0",
        dtype=torch.float,
    ):
        """
        :param Dynamics: nn module (object) that provides a forward function for propagating the dynamics forward in time
        :param Costs: nn module (object) that provides a forward function for evaluating the costs of state trajectories
        :param NX: state dimension
        :param CTRL_NOISE: (nu x nu) control noise covariance (assume v_t ~ N(u_t, CTRL_NOISE))
        :param device: pytorch device
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        """
        super(MPPI, self).__init__()
        self.data_logging = False
        self.Predmodel = Predmodel
        self.d = device
        self.dtype = torch.float
        self.K = MPPI_config["ROLLOUTS"]
        self.T = MPPI_config["TIMESTEPS"]
        self.M = MPPI_config["BINS"]
        self.u_per_command = MPPI_config["u_per_command"]

        self.Dynamics = Dynamics
        self.Costs = Costs
        self.Sampling = Sampling

        self.U = torch.zeros((self.T, self.Sampling.nu), dtype=self.dtype).to(self.d)


    @torch.jit.export
    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = torch.zeros((self.T, self.Sampling.nu), dtype=self.dtype).to(self.d)

    def get_action_samples(self, _state):
        self.U = torch.roll(self.U, self.u_per_command, dims=0)
        self.U[-self.u_per_command : , :] = self.U[-self.u_per_command,:] # repeat last control
        states = _state.view(1, -1).repeat(self.M, self.K, self.T, 1)
        controls, perturbation_cost = self.Sampling.sample(states, self.U)
        states = self.Dynamics.forward(states, controls)
        cost_total = torch.nan_to_num(
            self.Costs.forward(states, controls) + perturbation_cost, nan=1000.0
        )
        cost_total[:] = cost_total[0]
        controls, self.U = self.Sampling.update_control(cost_total, self.U, _state)

        return controls, states
    
    def forward(self, state):
        """
        :param: state
        :returns: best actions
        """
        ## shift command 1 time step
        self.U = torch.roll(self.U, self.u_per_command, dims=0)
        self.U[-self.u_per_command : , :] = self.U[-self.u_per_command,:] # repeat last control
        controls = self.optimize(state)
        return controls[:self.u_per_command]

    def optimize(self, _state):
        """
        :param: state
        :returns: best set of actions
        """
        ## sample perturbed actions
        states = _state.view(1, -1).repeat(self.M, self.K, self.T, 1)
        controls, perturbation_cost = self.Sampling.sample(states, self.U)
        ## All the states are initialized as copies of the current state
        ## M bins per control traj, K control trajectories, T timesteps, NX states
        ## update all the states using the dynamics function
        states = self.Dynamics.forward(states, controls)
        
        # # n_state, n_actions = self.dynpred_model.input_normalization(batch_state_.cuda(),batch_action_.cuda())
        if self.data_logging:
            cost_total = torch.nan_to_num(
                self.Costs.forward(states, controls) + perturbation_cost, nan=1000.0
            )
            controls, self.U = self.Sampling.update_control(cost_total, self.U, _state)            
            
        else:
            # n_controls = self.Predmodel.normalize_actions(controls)
            pred_outs = self.Predmodel(controls)

            cost_total = torch.nan_to_num(
                self.Costs.forward(states, controls) + perturbation_cost, nan=1000.0
            )
            controls, self.U = self.Sampling.update_control(cost_total, self.U, _state)            
            
            
           
        
        ## Evaluate costs on STATES with dimensions M x K x T x NX.
        ## Including the terminal costs in here is YOUR own responsibility!

        return controls