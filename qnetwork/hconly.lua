require 'nn'
require 'cunn'
require 'explore'
local matio = require 'matio'
local paths = require 'paths'
local load_data = require 'load_data'
local util = require 'util'

local model = '../models/debug1'
local config = '../config_onlyhc'

local fname_data = paths.concat(model, 'processed_data.mat')
local fname_saveresult = paths.concat(model, 'foo.png')
local fname_saveinit = paths.concat(model, 'init.png')
local fname_rlactions = paths.concat(config, 'rl_actions.txt')
local fname_stateids = paths.concat(config, 'rl_state_ids.txt')

rlacts = load_data.read_txt_table(fname_rlactions)
rlids = load_data.read_txt_table(fname_stateids)
pdata = matio.load(fname_data)
pdata.SARSA_list[{{},{#rlids+1}}]:apply(function (x) return x+1 end)
actsrl = util.rev_table(rlacts)
idsrl = util.rev_table(rlids)

-- Define Network
local net = nn.Sequential()
net:add(nn.Linear(3,30))
net:add(nn.ReLU())
net:add(nn.Linear(30,30))
net:add(nn.ReLU())
net:add(nn.Linear(30,30))
net:add(nn.ReLU())
net:add(nn.Linear(30,#rlacts))
--[[
net:get(1).weight:normal(0,0.01)
net:get(1).bias:fill(0)
net:get(3).weight:normal(0,0.01)
net:get(3).bias:fill(0)
net:get(5).weight:normal(0,0.01)
net:get(5).bias:fill(0)
--]]
net = net:cuda()

-- Define Criterion
local crit = nn.MSECriterion():cuda()

-- Define Dynamics Functions
local env = Dynamics(150)
env.width = pdata.voxel_grid:size(1)
env.length = pdata.voxel_grid:size(3)
env.height = pdata.voxel_grid:size(2)
env.epsilon = 0.3
local mkidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{idsrl['MakeHotChocolate']+1}}],actsrl['Do_MakeHotChocolate']))
local endidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{idsrl['MakeHotChocolate']+1}}],actsrl['Finish']))
mkidx = mkidx[1][1]; endidx = endidx[1][1]
env.hcpos = pdata.SARSA_list[{{mkidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.endpos = pdata.SARSA_list[{{endidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]

function env:query(state)
    qstate = torch.cdiv(state, torch.Tensor({{self.width, self.height, 1}}))*2-1
    return net:forward(qstate:cuda())
end

function env:new_state()
    local x = torch.random(0,self.width-1)
    local y = torch.random(0,self.length-1)
    local s = 0
    return torch.Tensor({{x,y,s}})
end

function env:reward(state, action, next_state)
    local x1 = state[1][idsrl['Pos_X']]
    local y1 = state[1][idsrl['Pos_Y']]
    local pos = torch.Tensor({{x1,y1}})
    local dist = torch.dist(pos,self.endpos)
    dist = 10*torch.exp(-dist)
    return dist
end

function env:transition(instate, action)
    local state = instate:clone()
    local a = rlacts[action]
    local x = state[1][idsrl['Pos_X']]
    local y = state[1][idsrl['Pos_Y']]
    local s = state[1][idsrl['MakeHotChocolate']]
    local isFinished = false

    if a == 'Move_North' then
        state[1][idsrl['Pos_Y']] = math.max(y-1, 0)
    elseif a == 'Move_South' then
        state[1][idsrl['Pos_Y']] = math.min(y+1, self.length-1)
    elseif a == 'Move_East' then
        state[1][idsrl['Pos_X']] = math.min(x+1, self.width-1)
    elseif a == 'Move_West' then
        state[1][idsrl['Pos_X']] = math.max(x-1, 0)
    elseif a == 'Do_MakeHotChocolate' then
        if s == 0 then
            state[1][idsrl['MakeHotChocolate']] = 1
        end
    elseif a == 'Finish' then
        if s == 1 then
            state[1][idsrl['MakeHotChocolate']] = 2
            isFinished = true
        end
    end
    return state, isFinished
end

function env:explore_action(state)
    local is_greed = torch.uniform() < self.epsilon
    local vec = self:query(state)
    local act = -1

    if is_greed then
        local n = actsrl['Move_North']
        local s = actsrl['Move_South']
        local e = actsrl['Move_East']
        local w = actsrl['Move_West']
        local h = actsrl['Do_MakeHotChocolate']
        local f = actsrl['Finish']
        local v = vec[1]
        local lu = {n,s,e,w,h,f}

        local opts = torch.Tensor({v[n],v[s],v[e],v[w],v[h],v[f]})
        _,gact = torch.max(opts, 1)
        act = lu[gact[1]]
        isFinish = 0
    else
        rset = {actsrl['Move_North'],
                actsrl['Move_South'],
                actsrl['Move_East'],
                actsrl['Move_West'],
                }
        if torch.uniform() < 0.9 then
            act = rset[torch.random(1,4)]
        elseif state[1][idsrl['MakeHotChocolate']] == 0 then
            act = actsrl['Do_MakeHotChocolate']
        elseif state[1][idsrl['MakeHotChocolate']] == 1 then
            act = actsrl['Finish']
        end
    end
    return act
end

-- Setup Export
local config = {}
config.learning_params = learning_params
config.net = net
config.crit = crit
config.env = env

return config
