import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import itertools

class PortfolioEnv(gym.Env):
    """
    A modular Gymnasium environment for Portfolio Management.
    
    Args:
        df (pd.DataFrame): The dataset containing dates and asset prices.
        reward_fn (callable): Function (env_instance) -> float.
        state_fn (callable): Function (env_instance) -> np.array.
        initial_amount (float): Starting cash.
        window_size (int): How many past timesteps to look at (if state_fn needs it).
        action_space_type (str): 'Continuous' or 'Discrete' for different approaches (Q-learning vs AC)
        state_space_type (str): 'Continuous' or 'Discrete' for different approaches (Q-learning vs AC)
        state_space_lim (float, float): inf and sup limits of the action space, can't be np.inf if Discrete
        n_bins (int): for discrete state space only, number of bins to map each stock weights in
        step_size (float): Discret action space only, size of the changes decided by the agent
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df,
                 reward_fn, 
                 state_fn, 
                 initial_amount=10000, 
                 window_size=1,
                 action_space_type='Continuous', # can be 'Discrete'
                 state_space_type='Continuous', # can be 'Discrete'
                 state_space_lim=(-np.inf, np.inf),
                 n_bins=10, # For discrete state space
                 step_size=0.05 # For discrete action space
                 ):
        super(PortfolioEnv, self).__init__()

        self.df = df.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Extract asset names (excluding Date)
        self.asset_names = [c for c in self.df.columns if c != 'Date']
        self.n_assets = len(self.asset_names)
        
        # Total dimensions: n_assets + 1 (for Cash/Not Invested)
        self.n_actions = self.n_assets + 1

        self.initial_amount = initial_amount
        self.window_size = window_size
        
        # Define reward function and state function (mapping to state space) from the passed arguments
        self.reward_fn = reward_fn
        self.state_fn = state_fn


        # Define Spaces

        # Store discretization params
        self.state_space_type = state_space_type
        self.state_space_lim = state_space_lim
        self.n_bins = n_bins
        self.step_size = step_size


        # Action: Weight vector (continuous).
        # Note: Agents can output raw scores (logits). We softmax them in step().
        # Define Action Space
        # In the discrete case, the work around is that we have for each stock 3 decision:(Sell, Do nothing, Buy) and this by
        # a fixed amount. This is the only solution to workaround the exploding dimensionality in the discrete case.
        if action_space_type == 'Discrete': # in this case we don't consider 
            self.action_map = list(itertools.product([-1, 0, 1], repeat=self.n_assets))
            self.action_space = spaces.Discrete(len(self.action_map))
            
            print(f"Discrete Mode: Created {len(self.action_map)} unique portfolio shift actions.")
        else:
            self.action_space = spaces.Box(low=-1, high=+1, shape=(self.n_actions,), dtype=np.float32)

        # State: Defined dynamically based on what the state_fn returns
        # We run a dummy reset to infer the shape
        self._dummy_reset()
        state_shape = self.get_state(raw=True).shape
        
        if self.state_space_type == 'Discrete':
            # MultiDiscrete expects an array of [n_bins, n_bins, ...] matching the state shape
            # We create a shape layout filled with 'n_bins'
            self.state_space = spaces.MultiDiscrete(np.full(state_shape, self.n_bins))
        else:
            self.state_space = spaces.Box(low=state_space_lim[0], high=state_space_lim[1], shape=state_shape, dtype=np.float32)

    def _dummy_reset(self):
        """Internal helper to set up dummy state for space initialization"""
        self.current_step = self.window_size
        self.portfolio_value = self.initial_amount
        self.weights = np.zeros(self.n_actions)
        self.weights[-1] = 1.0 # Start 100% in cash

    def _discretize(self, state):
        """
        Maps continuous state values into discrete bins.
        """
        low, high = self.state_space_lim
        
        if low == -np.inf or high == np.inf:
            raise ValueError("state_space_lim must be finite (e.g., (-1, 1)) when using 'Discrete' state space.")

        bins = np.linspace(low, high, self.n_bins + 1)
        
        digitized = np.digitize(state, bins) - 1
        
        return np.clip(digitized, 0, self.n_bins - 1)

    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            options = options if options is not None else {}

            # 1. Determine Start Step

            start_step = self.window_size # Default default

            if 'start_date' in options:
                # Find index of the specific date
                target_date = pd.to_datetime(options['start_date'])
                # Search for the date in the dataframe
                mask = self.df['Date'] == target_date
                if mask.any():
                    start_step = self.df.index[mask][0]
                else:
                    print(f"Warning: Date {options['start_date']} not found. Using default start.")
            
            elif 'start_index' in options:
                start_step = options['start_index']

            # Safety check: We cannot start before the window_size (need history)
            self.current_step = max(self.window_size, start_step)

            # 2. Determine End Step (Episode Length)
            # If episode_length is provided, we stop after N steps.
            # Otherwise, we go to the end of the DataFrame.
            if 'episode_length' in options:
                self.end_step = self.current_step + options['episode_length']
                # Clamp to max length of data
                self.end_step = min(self.end_step, len(self.df) - 1)
            else:
                self.end_step = len(self.df) - 1

            # 3. Initialize State
            self.portfolio_value = self.initial_amount
            
            # Initial weights: 100% Cash (last index)
            self.weights = np.zeros(self.n_actions)
            self.weights[-1] = 1.0 
            
            # History tracking
            self.history = {
                'portfolio_value': [self.initial_amount],
                'weights': [self.weights],
                'date': [self.df.iloc[self.current_step]['Date']]
            }

            return self.get_state(), {}

    def step(self, action):
        # Handle discrete action space
        if isinstance(self.action_space, spaces.Discrete):
            # Get the change vector for STOCKS only
            # e.g., [0, -1, 1] (Hold stock 0, Sell stock 1, Buy stock 2)
            deltas = np.array(self.action_map[action]) * self.step_size
            
            current_stock_weights = self.weights[:-1]
            
            new_stock_weights = current_stock_weights + deltas
            
            # Constraint: Cannot be negative (Short selling disabled)
            new_stock_weights = np.clip(new_stock_weights, 0, 1)
            
            # Cash = 100% - (Sum of all stock weights)
            sum_invested = np.sum(new_stock_weights)
            new_cash_weight = 1.0 - sum_invested
            
            # Handle "Overspending" (If sum_invested > 1.0)
            # If the agent wants to buy 60% Apple and 50% Google, it sums to 1.1
            # We must scale it down so Cash = 0 and Stocks sum to 1.0
            if new_cash_weight < 0:
                # Normalize stocks to sum exactly to 1.0
                new_stock_weights /= sum_invested
                new_cash_weight = 0.0
            
            self.weights = np.append(new_stock_weights, new_cash_weight)

        else:
            # Continuous Logic (Actor-Critic)
            action = np.exp(action) / np.sum(np.exp(action))
            self.weights = action

        # Simulate Evolution of the state space and calculate rewards:


        # Important Note !!!
        # We assume the prices are 'Close' prices
        # We use a Close-to-Close simplification
        # while quite standard in RL, it's not applicable in real life
        # issue: we use close price from day 1, and measure return as if we bought before day 1's close (which is impossible)
        # if we wanted to be perfectly realistic and deployable we would predict based on day 1's close, buy at day 2's open
        # and measure the return based on day 2's open vs day 2's close
        current_prices = self.df.iloc[self.current_step][self.asset_names].values
        next_prices = self.df.iloc[self.current_step + 1][self.asset_names].values

        # Calculate price relative vectors (price_t+1 / price_t)
        # Add the weight 1.0 for cash (we don't take into account inflation, maybe we should if we evaluate over long periods?)
        price_relatives = np.append(next_prices / current_prices, 1.0)

        # Calculate New Portfolio Value
        # Value_{t+1} = Value_t * dot(weights_t, price_relatives)
        step_return = np.sum(self.weights * price_relatives)
        self.portfolio_value *= step_return

        # The weights have shifted because some assets grew faster than others.
        # New Weight = (Old Weight * Price Relative) / Portfolio Return
        self.weights = (self.weights * price_relatives) / step_return
        
        # Increment Step
        self.current_step += 1
        # it's terminated when we went through all the data
        terminated = self.current_step >= self.end_step
        truncated = False

        # Calculate Reward (Modular)
        reward = self.reward_fn(self)

        # Update History
        self.history['portfolio_value'].append(self.portfolio_value)
        self.history['weights'].append(self.weights)
        self.history['date'].append(self.df.iloc[self.current_step]['Date'])

        return self.get_state(), reward, terminated, truncated, {}

    def get_state(self, raw=False):
        # Get the continuous state from the user-defined function
        state_value = self.state_fn(self)

        # If we want the Discrete version AND we aren't forcing raw output
        if self.state_space_type == 'Discrete' and not raw:
            return self._discretize(state_value)
            
        return state_value.astype(np.float32)