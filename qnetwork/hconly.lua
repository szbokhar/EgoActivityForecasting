require 'nn'
require 'cunn'
require 'explore'
local matio = require 'matio'
local paths = require 'paths'
local load_data = require 'load_data'
local util = require 'util'

local model = '../models/pdata3'
local config = '../config_onlyhc2'

local fname_data = paths.concat(model, 'processed_data.mat')
local fname_rlactions = paths.concat(config, 'rl_actions.txt')
local fname_stateids = paths.concat(config, 'rl_state_ids.txt')

rlacts = load_data.read_txt_table(fname_rlactions)
rlids = load_data.read_txt_table(fname_stateids)
pdata = matio.load(fname_data)
pdata.SARSA_list[{{},{#rlids+1}}]:apply(function (x) return x+1 end)
actsrl = util.rev_table(rlacts)
idsrl = util.rev_table(rlids)

input_size = #rlids
output_size = #rlacts

-- Define Network
local net = nn.Sequential()
net:add(nn.Linear(input_size,500))
net:add(nn.ReLU())
net:add(nn.Linear(500,1000))
net:add(nn.ReLU())
net:add(nn.Linear(1000,200))
net:add(nn.ReLU())
net:add(nn.Linear(200,output_size))
net = net:cuda()

-- Define Criterion
local crit = nn.MSECriterion():cuda()

-- Define Dynamics Functions
local env = Dynamics(150)
env.width = pdata.voxel_grid:size(1)
env.length = pdata.voxel_grid:size(3)
env.height = pdata.voxel_grid:size(2)
env.epsilon = 0.3
local mkidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Do_MakeHotChocolate']))
local endidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Finish']))
mkidx = mkidx[1][1]; endidx = endidx[1][1]
env.hcpos = pdata.SARSA_list[{{mkidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.endpos = pdata.SARSA_list[{{endidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.inSize = input_size
env.outSize = output_size

function env:query(state)
    local den = torch.ones(1,#rlids)
    den[1][idsrl['Pos_X']] = self.width
    den[1][idsrl['Pos_Y']] = self.height
    qstate = torch.cdiv(state, den)*2-1
    return net:forward(qstate:cuda())
end

function env:new_state()
    local x = torch.random(1,self.width)
    local y = torch.random(1,self.length)
    local st = torch.zeros(1,#rlids)
    st[1][idsrl['Pos_X']] = x
    st[1][idsrl['Pos_Y']] = y
    st[1][idsrl['BeforeHC']] = 1
    return st
end

function env:reward(state, action, next_state)
    local x1 = state[1][idsrl['Pos_X']]
    local y1 = state[1][idsrl['Pos_Y']]
    local bhc = state[1][idsrl['BeforeHC']]
    local ahc = state[1][idsrl['AfterHC']]
    local pos = torch.Tensor({{x1,y1}})

    local reward = 0
    --reward = reward + pdata.voxel_grid[{{x1},{6,12},{y1}}]:max()
    if bhc == 1 and action == actsrl['Do_MakeHotChocolate']then
        local dist = torch.dist(pos,self.hcpos)
        reward = reward - 50 * util.bool2int(dist>10)
    elseif ahc == 1 and action == actsrl['Finish'] then
        local dist = torch.dist(pos,self.endpos)
        reward = reward + 50 * util.bool2int(dist < 10)
    end
    return reward
end

function env:transition(instate, action)
    local state = instate:clone()
    local a = rlacts[action]
    local x = state[1][idsrl['Pos_X']]
    local y = state[1][idsrl['Pos_Y']]
    local bhc = state[1][idsrl['BeforeHC']]
    local ahc = state[1][idsrl['AfterHC']]
    local fin = state[1][idsrl['Finished']]
    local isFinished = false

    if a == 'Move_North' then
        state[1][idsrl['Pos_Y']] = math.max(y-1, 1)
    elseif a == 'Move_South' then
        state[1][idsrl['Pos_Y']] = math.min(y+1, self.length)
    elseif a == 'Move_East' then
        state[1][idsrl['Pos_X']] = math.min(x+1, self.width)
    elseif a == 'Move_West' then
        state[1][idsrl['Pos_X']] = math.max(x-1, 1)
    elseif a == 'Do_MakeHotChocolate' then
        if bhc == 1 then
            state[1][idsrl['BeforeHC']] = 0
            state[1][idsrl['AfterHC']] = 1
        end
    elseif a == 'Finish' then
        state[1][idsrl['BeforeHC']] = 0
        state[1][idsrl['AfterHC']] = 0
        state[1][idsrl['Finished']] = 1
        isFinished = true
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
    else
        rset = {actsrl['Move_North'],
                actsrl['Move_South'],
                actsrl['Move_East'],
                actsrl['Move_West'],
                }
        if torch.uniform() < 0.9 then
            act = rset[torch.random(1,4)]
        elseif state[1][idsrl['BeforeHC']] == 1 then
            act = actsrl['Do_MakeHotChocolate']
        elseif state[1][idsrl['AfterHC']] == 1 then
            act = actsrl['Finish']
        end
    end
    return act
end

function env:gen_grid(s)
    local w = self.width
    local h = self.length
    inn = torch.rand(w*h,#rlids):cuda()
    for x=0,(w-1) do
        for y=0,(h-1) do
            inn[1+x*h+y][idsrl['Pos_X']] = x
            inn[1+x*h+y][idsrl['Pos_Y']] = y
            inn[1+x*h+y][idsrl['BeforeHC']] = s[1]
            inn[1+x*h+y][idsrl['AfterHC']] = s[2]
            inn[1+x*h+y][idsrl['Finished']] = s[3]
        end
    end
    inn = self:normalize_points(inn)
    return inn
end

function env:normalize_points(pts)
    local w = self.width
    local h = self.length
    pts[{{},{idsrl['Pos_X']}}] = pts[{{},{idsrl['Pos_X']}}]*2/w-1
    pts[{{},{idsrl['Pos_Y']}}] = pts[{{},{idsrl['Pos_Y']}}]*2/h-1

    return pts
end


-- Setup Export
local config = {}
config.learning_params = learning_params
config.net = net
config.crit = crit
config.env = env

return config
