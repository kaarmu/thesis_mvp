from do_mpc.controller import MPC
from do_mpc.model import Model


class ModelPredictiveController(MPC):

    def __init__(self,
                 model: Model,
                 *,
                 step: float,
                 horizon: int,
                 weights: dict,
                 constraints: dict,
                 **kwargs):

        super().__init__(model=model, **kwargs)

        self.set_param(
            n_robust=0,
            n_horizon=horizon,
            t_step=step,
            store_full_solution=True,
            nlpsol_opts = {
                # suppress output
                'ipopt.print_level': 0,
                'ipopt.sb': 'yes',
                'print_time': 0,
            },
        )

        ## OBJECTIVE ##

        state = model.x
        wi = weights['input']
        ws = weights['state']
        wt = weights['terminal']

        # Error cost
        lterm = (ws['x'] * state['x']**2 +
                 ws['y'] * state['y']**2 +
                 ws['v'] * state['v']**2 +
                 ws['yaw'] * state['yaw']**2)

        # Terminal cost
        mterm = (wt['x'] * state['x']**2 +
                 wt['y'] * state['y']**2 +
                 wt['v'] * state['v']**2 +
                 wt['yaw'] * state['yaw']**2)

        self.set_objective(lterm=lterm, mterm=mterm)
        self.set_rterm(**wi)

        ## CONSTRAINTS ##

        for direction in ('lower', 'upper'):

            # States
            for name in ('x', 'y', 'v', 'yaw'):
                if (direction, name) in constraints:
                    self.bounds[direction, '_x', name] = constraints[direction, name]

            # inputs
            for name in ('steering', 'velocity'):
                if (direction, name) in constraints:
                    self.bounds[direction, '_u', name] = constraints[direction, name]

        ## BUILD CONTROLLER ##

        self.setup()

