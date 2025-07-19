# LQR-and-MPC-optimal-trajectory-tracking
Using the Linear Quadratic Regulator (LQR) and Model Predictive Control (MPC) to optimally track system's trajectories derived by Newton's method

The objective is to generate a desired optimal trajectory and follow it properly using both methods in Python environment. In the project, the desired curve's transitions between two system equilibria are connected by a 5\textsuperscript{th}-order polynomial and optimized via Newton's method. The obtained optimal trajectory has been successfully tracked by both LQR and MPC methods, achieving close to zero tracking error within small time for perturbed initial conditions. The animation in Python has clearly demonstrated the smooth transitions of the flexible surface between its equilibrium points. 
