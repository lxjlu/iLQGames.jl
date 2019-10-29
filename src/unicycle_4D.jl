"""
$(TYPEDEF)

A simple feedback linearizable unicycle model.

# Fields

$(TYPEDFIELDS)
"""
# TODO: maybe use the
struct Unicycle4D{ΔT} <: ControlSystem{ΔT, 4, 2} end

function dx(cs::Unicycle4D, x::SVector{4}, u::SVector{2}, t::AbstractFloat)
    # position: x
    dx1 = x[4] * cos(x[3])
    # position: y
    dx2 = x[4] * sin(x[3])
    # orientation
    dx3 = u[1]
    # velocity
    dx4 = u[2]

    return @SVector[dx1, dx2, dx3, dx4]
end

function linearize_discrete(cs::Unicycle4D, x::SVector{4}, u::SVector{2},
                            t::AbstractFloat)
    ΔT = samplingtime(cs)
    cθ = cos(x[3]) * ΔT
    sθ = sin(x[3]) * ΔT

    A = @SMatrix [1 0 -x[4]*sθ cθ;
                  0 1 x[4]*cθ  sθ;
                  0 0 1        0;
                  0 0 0        1]

    B = @SMatrix [0  0;
                  0  0;
                  ΔT 0;
                  0  ΔT];
    return LinearSystem{ΔT}(A, B)
end

xyindex(cs::Unicycle4D) = @SVector [1, 2]

"------------------- Implement Feedback Linearization Interface -------------------"

# TODO: Maybe use StaticArrays.FieldVector here to have a type safe desctiction
# between ξ and x coordinates.

function feedback_linearized_system(cs::Unicycle4D)
    ΔT = samplingtime(cs)
    # ξ = (px, pẋ, py, pẏ)
    A = @SMatrix [1. ΔT 0. 0.;
                  0. 1. 0. 0.;
                  0. 0. 1. ΔT;
                  0. 0. 0. 1.;]
    B = @SMatrix [0. 0.;
                  ΔT 0.;
                  0. 0.;
                  0. ΔT];
    return LinearSystem{ΔT}(A, B)
end

function x_from(cs::Unicycle4D, ξ::SVector{4})
    # px, py, θ, v
    return @SVector [ξ[1], ξ[3], atan(ξ[4], ξ[2]), sqrt(ξ[2]^2 + ξ[4]^2)]
end

function ξ_from(cs::Unicycle4D, x::SVector{4})
    # px, pẋ, py, pẏ
    return @SVector [x[1], cos(x[3])*x[4], x[2], sin(x[3])*x[4]]
end

function λ_issingular(cs::Unicycle4D, ξ::SVector{4})
    vx = ξ[2]; vy = ξ[4]
    ϵ = 0.01
    # cant invert for velocities near 0
    return isnan(vx) || isnan(vy) || (abs(vx) < ϵ && abs(vy) < ϵ)
end