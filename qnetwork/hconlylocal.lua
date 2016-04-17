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
pdata.voxel_grid = pdata.voxel_grid:cuda()

input_size = #rlids
output_size = #rlacts

-- Define Network
local net = nn.Sequential()
--net:add(nn.Linear(input_size,4096))
--[[
net:add(nn.Linear(input_size-3,1000))
net:add(nn.ReLU())
net:add(nn.Linear(1000,1000))
net:add(nn.ReLU())
net:add(nn.Linear(1000,output_size))
--]]
net:add(nn.SpatialConvolution(1,20,5,5,1,1,2,2))
net:add(nn.ReLU())
net:add(nn.View(81*20))
net:add(nn.Linear(81*20,1000))
net:add(nn.ReLU())
net:add(nn.Linear(1000,9))
net = net:cuda()

-- Define Criterion
local crit = nn.MSECriterion():cuda()

-- Define Dynamics Functions
local env = Dynamics(150)
env.width = pdata.voxel_grid:size(1)
env.length = pdata.voxel_grid:size(3)
env.height = pdata.voxel_grid:size(2)
print(pdata.voxel_grid:size())
env.epsilon = 0.3
local mkidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Do_MakeHotChocolate']))
local endidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Finish']))
mkidx = mkidx[1][1]; endidx = endidx[1][1]
env.hcpos = pdata.SARSA_list[{{mkidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.endpos = pdata.SARSA_list[{{endidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.inSize = input_size
env.outSize = output_size
env.grid2dpad = 4
env.grid2d = pdata.voxel_grid[{{},{6,12},{}}]:max(2):view(env.width, env.length)
env.grid2d = env.grid2d - env.grid2d:mean()
env.grid2d = env.grid2d / env.grid2d:std()
env.grid2d = nn.SpatialZeroPadding(env.grid2dpad,
        env.grid2dpad, env.grid2dpad, env.grid2dpad):forward(env.grid2d:view(1,env.width, env.length))
        :view(env.width+2*env.grid2dpad, env.length+2*env.grid2dpad)

function env:query(state)
    qstate = self:normalize_points(state)
    net:forward(qstate)
    return net.output
end

function env:new_state()
    local x = torch.random(1,self.width)
    local y = torch.random(1,self.length)
    --local st = torch.zeros(1,#rlids)
    local st = torch.zeros(1,#rlids-3)
    st[1][idsrl['Pos_X']] = x
    st[1][idsrl['Pos_Y']] = y
    --st[1][idsrl['BeforeHC']] = 1
    return st
end

function env:reward(state, action, next_state)
    local x1 = state[1][idsrl['Pos_X']]
    local y1 = state[1][idsrl['Pos_Y']]
    --local bhc = state[1][idsrl['BeforeHC']]
    --local ahc = state[1][idsrl['AfterHC']]
    local pos = torch.Tensor({{x1,y1}})

    local reward = 0
    --[[
    if bhc == 1 and action == actsrl['Do_MakeHotChocolate']then
        local dist = torch.dist(pos,self.hcpos)
        reward = reward - 50 * util.bool2int(dist>5)
    elseif ahc == 1 and action == actsrl['Finish'] then
        local dist = torch.dist(pos,self.endpos)
        reward = reward + 50 * util.bool2int(dist < 5)
    end
    --]]
    if action == actsrl['Do_MakeHotChocolate']then
        local dist = torch.dist(pos,self.hcpos)
        reward = reward + 50 * util.bool2int(dist<4)
        --[[
    else
        reward = reward - 0.5*pdata.voxel_grid[{{x1},{6,12},{y1}}]:max()
        --]]
    end
    return reward
end

function env:transition(instate, action)
    local state = instate:clone()
    local a = rlacts[action]
    local x = state[1][idsrl['Pos_X']]
    local y = state[1][idsrl['Pos_Y']]
    --local bhc = state[1][idsrl['BeforeHC']]
    --local ahc = state[1][idsrl['AfterHC']]
    --local fin = state[1][idsrl['Finished']]
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
        isFinished = true
        --[[
        if bhc == 1 then
            state[1][idsrl['BeforeHC'] ] = 0
            state[1][idsrl['AfterHC'] ] = 1
            state[1][idsrl['Finished'] ] = 0
        end
    elseif a == 'Finish' then
        state[1][idsrl['BeforeHC'] ] = 0
        state[1][idsrl['AfterHC'] ] = 0
        state[1][idsrl['Finished'] ] = 1
        isFinished = true
        --]]
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
        local v = vec
        local lu = {n,s,e,w,h,f}

        local opts = torch.Tensor({v[n],v[s],v[e],v[w],v[h]})
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
        else
            act = actsrl['Do_MakeHotChocolate']
            --[[
        elseif state[1][idsrl['BeforeHC'] ] == 1 then
            act = actsrl['Do_MakeHotChocolate']
        elseif state[1][idsrl['AfterHC'] ] == 1 then
            act = actsrl['Finish']
            --]]
        end
    end
    return act
end

function env:gen_grid(s)
    local w = self.width
    local h = self.length
    --inn = torch.rand(w*h,#rlids):cuda()
    inn = torch.Tensor(w*h,#rlids-3):cuda()
    for x=1,w do
        for y=1,h do
            inn[1+(x-1)*h+y-1][idsrl['Pos_X']] = x
            inn[1+(x-1)*h+y-1][idsrl['Pos_Y']] = y
        end
    end
    inn = self:normalize_points(inn)
    return inn
end

function env:normalize_points(pts)
    local w = self.width
    local h = self.length
    --[[
    pts[{{},{idsrl['Pos_X']}}] = pts[{{},{idsrl['Pos_X']}}]*2/w-1
    pts[{{},{idsrl['Pos_Y']}}] = pts[{{},{idsrl['Pos_Y']}}]*2/h-1
    --]]

    local ret = torch.Tensor(pts:size(1),1,9,9):cuda()

    for i=1,pts:size(1) do
        local x = pts[i][1]+env.grid2dpad
        local y = pts[i][2]+env.grid2dpad
        local map = self.grid2d[{{x-4,x+4},{y-4,y+4}}]
        ret[i][1] = map
    end

    return ret
end


-- Setup Export
local config = {}
config.learning_params = learning_params
config.net = net
config.crit = crit
config.env = env

return config
