using iLQGames
import iLQGames: dx

# parameters: number of states, number of inputs, sampling time, horizon
nx, nu, ΔT, game_horizon = 4, 2, 0.1, 200

# setup the dynamics
struct Unicycle <: ControlSystem{ΔT,nx,nu} end
# the differential equation of a uncycle with state: (px, py, phi, v)
dx(cs::Unicycle, x, u, t) = SVector(x[4]cos(x[3]), x[4]sin(x[3]), u[1], u[2])
dynamics = Unicycle()


# player-1 wants the unicycle to stay close to the origin,
# player-2 wants to keep close to 1 m/s
costs = (FunctionPlayerCost((g, x, u, t) -> (x[1]^2 + x[2]^2 + u[1]^2)),
         FunctionPlayerCost((g, x, u, t) -> ((x[4] - 1)^2 + u[2]^2)))
# indices of inputs that each player controls
player_inputs = (SVector(1), SVector(2))


g = GeneralGame(game_horizon, player_inputs, dynamics, costs)


# get a solver, choose initial conditions and solve (in about 9 ms with automatic differentiation)
solver = iLQSolver(g)
x0 = SVector(1, 1, 0, 0.5)
converged, trajectory, strategies = solve(g, solver, x0)


# for visualization we need to state which state indices correspond to px and py
position_indices = tuple(SVector(1,2))
# Note: you can use the `plot_traj` call without @animated to get a non-animated plot instead.
@animated(plot_traj(trajectory, g, [:red, :green], player_inputs),
          1:game_horizon, "minimal_example.gif")
