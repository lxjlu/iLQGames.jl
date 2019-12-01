struct NPlayerUnicycleCost{nx,nu,TXI<:NTuple,TUI<:NTuple,TIC,TICR,TSC,TSCR,TPC,TOC,TGC}<:NPlayerNavigationCost{nx,nu}
    "the index of the player this cost applies to"
    player_id::Int
    xids::TXI
    uids::TUI
    inputcost::TIC
    inputconstr::TICR
    statecost::TSC
    stateconstr::TSCR
    proximitycost::TPC
    obstaclecost::TOC
    goalcost::TGC
end

function NPlayerUnicycleCost(player_id, xids::TXI, uids::TUI, xg, t_final;
         inputcost::TIC=QuadCost(SMatrix{2,2}([5. 0.; 0. 2.]) * 10),
         inputconstr::TICR=tuple(),
         statecost::TSC=QuadCost(SMatrix{4,4}(diagm([0., 0., 0., 1.])) * 50),
         stateconstr::TSCR=(SoftConstr(2, -0.75, 0.75, 50),), # wall constraint
         proximitycost::TPC=ProximityCost(1.0, 50),
         obstaclecost::TOC=tuple(),
         goalcost::TGC=GoalCost(t_final, xg, SMatrix{4,4}(diagm([1.,1.,1.,0.])) *
                                300)) where {TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TOC,TGC}

    nx = sum(length.(xids))
    nu = sum(length.(uids))
    return NPlayerUnicycleCost{nx,nu,TXI,TUI,TIC,TICR,TSC,TSCR,TPC,TOC,TGC}(
        player_id, xids, uids, inputcost, inputconstr, statecost, stateconstr,
        proximitycost, obstaclecost, goalcost)
end

"-----------------Implementing the NPlayerNavigationCost interface-----------------"

player_id(c::NPlayerUnicycleCost) = c.player_id
inputcost(c::NPlayerUnicycleCost) = c.inputcost
inputconstr(c::NPlayerUnicycleCost) = c.inputconstr
statecost(c::NPlayerUnicycleCost) = c.statecost
stateconstr(c::NPlayerUnicycleCost) = c.stateconstr
proximitycost(c::NPlayerUnicycleCost) = c.proximitycost
obstaclecost(c::NPlayerUnicycleCost) = c.obstaclecost
goalcost(c::NPlayerUnicycleCost) = c.goalcost
xindex(pc::NPlayerUnicycleCost) = pc.xids
uindex(pc::NPlayerUnicycleCost) = pc.uids

"------------------ Implementing Feedback Linearization Interface -----------------"

"Here, we don't transform the cost explicitly but as suggested in
https://arxiv.org/abs/1910.00681 we formulate a new cost directly in ξ coordinates."
function transformed_cost(cs::Unicycle4D, c::NPlayerUnicycleCost)
    @unpack t_active, xg = goalcost(c)
    ξg = ξ_from(cs, xg)
    if λ_issingular(cs, ξg)
        @warn "State conversion map is singular at provided goal state."
    end
    return NPlayer2DDoubleIntegratorCost(player_id(c), xindex(c), uindex(c), ξg,
                                         t_active)
end
