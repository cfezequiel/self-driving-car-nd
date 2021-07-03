## P8: PID control

<p align="center">
  <img src="./images/pid_control.gif"><br>
  <em>PID controller-driven car in simulated track.</em>
</p>



## Objective

Implement a PID controller in C++ to enable a virtual car to drive around a simulated track.

## Usage

Check out the Udacity [project repo](https://github.com/udacity/CarND-PID-Control-Project) on path planning and copy (overwrite) the following files there:

-  [`src/`](./src)
- [`CMakeLists.txt`](./CMakeLists.txt)

Refer to the corresponding Udacity project's [README](https://github.com/udacity/CarND-PID-Control-Project/blob/master/README.md) for instructions on how to compile and run the code in the simulator.

The enable parameter tuning of the PID gains, uncomment the line containing `#define USE_TWIDDLE 1` in [`main.cpp`](./src/main.cpp).

## Rubric points

### Compilation

The code should compile without errors using `cmake` and `make` with the files in this repo.

### Implementation

#### PID controller

The PID controller implementation, found in [PID.cpp](./src/PID.cpp),  follows closely with that described in the relevant lesson on PID control. The `PID` class implements the PID controller. The `Init` method of the class initializes the P, I and D params (using `UpdateParams` method) and also sets the `i_error` to zero and `is_first_cte` flag to `true`. The `i_error` variable accumulates errors (`cte`) for the Integral part of the PID. The `is_first_cte` flag is used to perform behaviors during the initial `cte` read in `UpdateError`.

The `UpdateError` method updates the `p_error`, `i_error` and `d_error` attributes update the proportional, integral and derivative errors respectively. The `TotalError` method then returns the sum of each P, I and D error multiplied by its coefficient, `Kp`, `Ki` and `Kd` respectively.

#### Parameter tuning

The `Kp`, `Ki` and `Kd` parameters of the PID controller were tuned using the twiddle algorithm explained in the PID lesson. A succinct Python implementation can be found in [1] which was used as a reference for the C++ implementation. However, the algorithm needed to be altered to account for the fact that the parameter updates would be made after each telemetry event (as opposed to each simulation run).

The `Twiddle` class found in [`twiddle.cpp`](./src/twiddle.cpp) implements the algorithm similar to the Python example in [1]. The constructor initializes the target parameters to tune (`p`), the change in each parameter (`dp`) and a few other housekeeping variables. The `pni` flag checks whether the previous parameter update resulted in no change, `num_params` simply stores the number of target params, `best_err_set` checks if the best error value has been set (this is false initially), the `threshold` for suppressing further parameter updates and the target parameter to update given by `param_idx`.

The `UpdateNext` method determines which target parameter to update next, which basically cycles through `p`. The selected parameter is then updated using its corresponding `dp` value.

The `Update` method updates `p` and `dp` according the twiddle formulation. Initially, the best error is set to whatever error value (`err`) that is passed and the first target parameter is selected. If the sum of all `dp` values is less than `threshold` (meaning the error change is too small), no further parameter updates are performed. Otherwise, parameters are updated depending on whether the current error exceeds the best error seen. 

A modification to the algorithm in [1] involves determining whether the previous run of the simulation resulted in no improvement to the best error, which is determined by the `pni` attribute. 

#### Execution loop

In `main.cpp`, the PID class is instantiated, and the errors are updated on every telemetry event. The PID controller is currently used to affect car steering only, where the `steer_value` is equal to the negated `TotalError` value. The `steer_value` is set to be within the range [-1, 1].

The speed of the car is controlled by the `throttle` parameter, which is updated based on `MIN_SPEED` and `MAX_SPEED` thresholds, `steer_value` and the current speed, according to this formula found in [2]:

```cpp
speed_limiter = speed > speed_limiter ? MIN_SPEED : MAX_SPEED;
double throttle = (
    1.0
    - pow(steer_value, 2)
    - pow(speed/speed_limiter, 2));
```

where the `throttle` value is set to be within the range [0, 1]. The higher the `steer_value`, the lower the throttle. Similarly, the greater the `speed`/`speed_limiter` ratio, the lower the throttle.

Parameter tuning is also performed in `main.cpp`, but is not done on every telemetry event. Instead, each telemetry event updates a counter `count` and the update is done when count reaches the `TWIDDLE_BATCH_SIZE` value. The error used (`batch_error`) for twiddle is the average of the absolute `cte` accumulated in `TWIDDLE_BATCH_SIZE` runs (`total_error / count`). After the parameters are updated, the `total_error` is set to zero.

Since using Twiddle to update the PID parameters will result in erratic behavior of the virtual car and may lead to a crash, the initial PID parameters (`params`) are set to some "good" values taken from [3]. The twiddle update is then more of a refinement of the initial parameters. The initial parameter changes `dparams` are set in proportion to `params`, taking 1/10th of the values of the latter.

Below shows the change in parameters using twiddle after a few laps around the track.

| Parameter      | Initial value | Final value  |
| :------------  | :----------:  | -----------: |
| Kp             | 1.50e-01      | 1.81e-01     |
| Ki             | 4.00e-05      | 4.34e-05     |
| Kd             | 3.00e+00      | 3.00e+00     |

### Reflection

To study the effects of proportional, integral and derivative components of the PID controller, I conducted an ablation study, removing one or two components by of the PID error calculation by  zeroing out the relevant PID coefficients `Kp`, `Ki` and `Kd` respectively. I ran the experiments using the final values for  `Kp`, `Ki` and `Kd` and disabled twiddle parameter tuning during the simulations.

#### P-only

Using only the proportional component, the controller will prioritize getting to the target value as fast as possible and would give a steer value proportional to the error, but may overshoot and oscillate around the target (lane center) depending on the `Kp` value. [This video](https://youtu.be/jeFriZ1xjlg) demonstrates this behavior. At some point, the virtual car gets stuck in the curb due to the oscillations.

#### I-only

The integral component is used to remove any steady state error. Since the error is accumulated, a steady state error will eventually force the controller to output a steering value that would be enough to force the virtual car to move towards the desired reference state. In the simulation, only a small contribution from I was needed in order to stabilize the virtual car, and hence the `Ki` value was kept small. [This video](https://youtu.be/fOsbKjs1MNE) shows the behavior of the virtual car in simulation using only the I component of the controller. The car did not turn very much, likely due to the small `Ki` value resulting in small steering values and tended to move straight, eventually veering off the track which was turning to the left.

#### D-only

Used in conjunction with the P component, the derivative (D) component helps to balance out the former's tendency to overshoot the target by taking into account the change in error (usually negative) when determining the steering value. This tends to reduce the output steering value which would otherwise be greater using the P controller alone. 

[This video](https://youtu.be/kdNP5ENOe-4) shoes the behavior of the virtual car with only a D controller. While the track is veering left, the controller is able to steer the car enough to keep it on the track up to a certain point, since the increasing error results in a negative steer value that moves the car towards the left. However, the amount of steer is not enough to move the car back to center lane and eventually it goes off track at the steeper turn before the bridge.

#### PD-only

[This video](https://youtu.be/ObHjwy74A0Y) shows the behavior of the virtual car using a PD controller. The controller is able to correct the steer value so that the car can stay in the center of the lane. Comparing the performance with the P-only scenario, it can be seen that the D component is able to help avoid the oscillating behavior shown in the former. The car is able to navigate through the steep left turn before the bridge, but the steer values across updates may need to be smoothened out more to generate smoother turns.

### Simulation

[This video](https://youtu.be/Pda6FVRmxiE) shows the performance of the virtual car using a PID controller. The controller is able to navigate the car around the simulated track the car leaving the drivable portion of the track surface.




## References

[1] https://martin-thoma.com/twiddle/

[2] https://github.com/naokishibuya/car-behavioral-cloning/blob/master/drive.py

[3] https://github.com/justinlee007/CarND-PID-Control-Project/blob/master/src/main.cpp

[4] [Simple Examples of PID Control (YouTube video)](https://www.youtube.com/watch?v=XfAt6hNV8XM)



