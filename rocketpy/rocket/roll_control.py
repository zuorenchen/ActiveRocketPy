import warnings

import numpy as np

from ..prints.roll_control_prints import _RollControlPrints


class RollControl:
    """Roll Control system class for managing rocket roll torque.

    This class represents a roll control system that allows the application
    of roll torque around the rocket's X-axis. Ideal roll torque is assumed.

    Attributes
    ----------
    RollControl.roll_torque : float
        Current roll torque magnitude in N·m (Newton-meters).
        Positive values indicate counter-clockwise rotation when viewed
        from the nose of the rocket.
    RollControl.max_roll_torque : float
        Maximum roll torque magnitude in N·m. The roll torque is clamped
        to this value if clamp is True.
    RollControl.clamp : bool, optional
        If True, roll torque is clamped to [-max_roll_torque, max_roll_torque].
        If False, a warning is issued when roll torque exceeds the max value.
    RollControl.name : str
        Name of the roll control system.
    """

    def __init__(
        self,
        sampling_rate=100,
        max_roll_torque=0,
        torque_rate_limit=0,
        clamp=True,
        roll_torque=0.0,
        name="Roll Control",
        actuator_tau=None,
    ):
        """Initializes the RollControl class.

        Parameters
        ----------
        sampling_rate : int, optional
            Sampling rate of the roll control system in Hz. Default is 100 Hz.
        max_roll_torque : float, int
            Maximum roll torque magnitude in N·m. Must be non-negative.
            Default is 0 (no roll control).
        torque_rate_limit : float, int
            Maximum roll torque rate in N·m/s. Roll torque is limited to this
            rate. Must be non-negative. Default is 0 (no torque change).
        clamp : bool, optional
            If True, the simulation will clamp roll torque to the range
            [-max_roll_torque, max_roll_torque] if it exceeds this range.
            If False, the simulation will issue a warning if roll torque
            exceeds the maximum value. Default is True.
        roll_torque : float, optional
            Initial roll torque in N·m. Default is 0.0 (no torque).
        name : str, optional
            Name of the roll control system. Default is "Roll Control".
        actuator_tau : float, optional
            Time constant for the torque actuator dynamics (first-order IIR
            filter) in seconds. If None, no actuator dynamics are applied.
            Must be non-negative. Default is None.

        Returns
        -------
        None
        """
        self.name = name
        self.sampling_rate = sampling_rate
        assert max_roll_torque >= 0, "max_roll_torque must be non-negative."
        self.max_roll_torque = max_roll_torque
        assert torque_rate_limit >= 0, "torque_rate_limit must be non-negative."
        self.torque_rate_limit = torque_rate_limit
        self.clamp = clamp
        # Actuator dynamics parameters
        if actuator_tau is not None:
            assert actuator_tau >= 0, "actuator_tau must be non-negative."
        self.actuator_tau = actuator_tau
        # Compute IIR filter coefficients (first-order system)
        self._update_iir_coefficients()
        self.initial_roll_torque = roll_torque
        self.roll_torque_prev = roll_torque
        self.roll_torque_filtered = roll_torque
        self.roll_torque = roll_torque
        self.prints = _RollControlPrints(self)

    def _update_iir_coefficients(self):
        """Updates the IIR filter coefficient based on time constant and
        sampling rate. Uses first-order discrete-time system:
        y[n] = alpha * u[n] + (1 - alpha) * y[n-1]
        where alpha = Ts / (tau + Ts)
        """
        sampling_period = 1.0 / self.sampling_rate
        if self.actuator_tau is not None and self.actuator_tau > 0:
            self._alpha = sampling_period / (self.actuator_tau + sampling_period)
        else:
            self._alpha = 1.0  # No filtering, direct pass-through

    @property
    def roll_torque(self):
        """Returns the current roll torque in N·m."""
        return self._roll_torque

    @roll_torque.setter
    def roll_torque(self, value):
        """Sets the roll torque with optional clamping or warning.

        Parameters
        ----------
        value : float
            Roll torque in N·m.
        """
        if abs(value) > self.max_roll_torque:
            if self.clamp:
                value = np.clip(value, -self.max_roll_torque, self.max_roll_torque)
            else:
                warnings.warn(
                    f"Roll torque of {self.name} is {value:.4f} N·m, "
                    f"which exceeds the maximum of {self.max_roll_torque:.4f} N·m.",
                    UserWarning,
                )
        # Limit the roll torque rate
        max_torque_change = self.torque_rate_limit / self.sampling_rate
        torque_change = value - self.roll_torque_prev
        if abs(torque_change) > max_torque_change:
            value = self.roll_torque_prev + np.sign(torque_change) * max_torque_change
        self.roll_torque_prev = value
        # Apply first-order IIR actuator dynamics
        self.roll_torque_filtered = (
            self._alpha * value + (1 - self._alpha) * self.roll_torque_filtered
        )
        self._roll_torque = self.roll_torque_filtered

    def _reset(self):
        """Resets the roll control system to its initial state. This method
        is called at the beginning of each simulation to ensure the roll
        control system is in the correct state."""
        self.roll_torque_prev = self.initial_roll_torque
        self.roll_torque_filtered = self.initial_roll_torque
        self.roll_torque = self.initial_roll_torque

    def info(self):
        """Prints summarized information of the roll control system.

        Returns
        -------
        None
        """
        self.prints.basics()

    def all_info(self):
        """Prints all information of the roll control system.

        Returns
        -------
        None
        """
        self.info()

    def to_dict(self, **kwargs):  # pylint: disable=unused-argument
        return {
            "sampling_rate": self.sampling_rate,
            "max_roll_torque": self.max_roll_torque,
            "torque_rate_limit": self.torque_rate_limit,
            "clamp": self.clamp,
            "roll_torque": self.initial_roll_torque,
            "actuator_tau": self.actuator_tau,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            sampling_rate=data.get("sampling_rate"),
            max_roll_torque=data.get("max_roll_torque"),
            torque_rate_limit=data.get("torque_rate_limit"),
            clamp=data.get("clamp"),
            roll_torque=data.get("roll_torque"),
            actuator_tau=data.get("actuator_tau"),
            name=data.get("name"),
        )
