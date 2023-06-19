import casadi as ca
from do_mpc.model import Model


class BicycleModel(Model):

    def __init__(self,
                 base_length: float = 0.32,     # [m]
                 esc_gain: float = 0.1,
                 **kwargs):

        super().__init__(model_type='continuous', **kwargs)

        ## STATE VARIABLES ##

        x = self.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        y = self.set_variable(var_type='_x', var_name='y', shape=(1, 1))
        v = self.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        yaw = self.set_variable(var_type='_x', var_name='yaw', shape=(1, 1))

        ## INPUT VARIABLES ##

        steering = self.set_variable(var_type='_u', var_name='steering', shape=(1, 1))
        velocity = self.set_variable(var_type='_u', var_name='velocity', shape=(1, 1))

        ## STATE SPACE DYNAMICS ##

        self.set_rhs('x', v * ca.cos(yaw))
        self.set_rhs('y', v * ca.sin(yaw))
        self.set_rhs('v', v/base_length * ca.tan(steering))
        self.set_rhs('yaw', 1/esc_gain * (velocity - v))

        ## BUILD MODEL ##

        self.setup()

