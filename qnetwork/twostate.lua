require 'nn'
require 'cunn'
require 'explore'
local matio = require 'matio'
local paths = require 'paths'
local load_data = require 'load_data'
local util = require 'util'

local model = '../models/3state_crop_data'
local config = '../config_2state'

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

input_size = #rlids-1
output_size = #rlacts
print(output_size)

-- Define Network

local net1 = nn.Sequential()
net1:add(nn.Linear(input_size,1000))
net1:add(nn.ReLU())
net1:add(nn.Linear(1000,1000))
net1:add(nn.ReLU())
net1:add(nn.Linear(1000,output_size))
net1 = net1:cuda()
local net2 = nn.Sequential()
net2:add(nn.Linear(input_size,1000))
net2:add(nn.ReLU())
net2:add(nn.Linear(1000,1000))
net2:add(nn.ReLU())
net2:add(nn.Linear(1000,output_size))
net2 = net2:cuda()
local net3 = nn.Sequential()
net3:add(nn.Linear(input_size,1000))
net3:add(nn.ReLU())
net3:add(nn.Linear(1000,1000))
net3:add(nn.ReLU())
net3:add(nn.Linear(1000,output_size))
net3 = net3:cuda()
local net4 = nn.Sequential()
net4:add(nn.Linear(input_size,1000))
net4:add(nn.ReLU())
net4:add(nn.Linear(1000,1000))
net4:add(nn.ReLU())
net4:add(nn.Linear(1000,output_size))
net4 = net4:cuda()

local allNets = {}
allNets[1] = net1
allNets[2] = net2
allNets[3] = net3
allNets[4] = net4

function netClone(self)
    local newall = {}
    newall[1] = self[1]:clone()
    newall[2] = self[2]:clone()
    newall[3] = self[3]:clone()
    newall[4] = self[4]:clone()
    return newall
end

function netZeroGradParameters(self)
    self[1]:zeroGradParameters()
    self[2]:zeroGradParameters()
    self[3]:zeroGradParameters()
    self[4]:zeroGradParameters()
end

function netForward(net, batchIn)
    batchOut = torch.zeros(batchIn:size(1), output_size)

    for i=1,#net do
        local miniIdx = batchIn[{{},{3}}]:eq(i):nonzero()
        if miniIdx:nDimension() ~= 0 then
            miniIdx = miniIdx:select(2,1)
            local miniBatch = batchIn:index(1,miniIdx)[{{},{1,2}}]:cuda()
            local cval = net[i]:forward(miniBatch):type('torch.DoubleTensor')
            fidx = 1
            miniIdx:apply(function(x) batchOut[x] = cval[fidx]; fidx=fidx+1; return x; end)
        end
    end

    return batchOut
end

function netBackward(net, batchIn, grad)
    for i=1,#net do
        local miniIdx = batchIn[{{},{3}}]:eq(i):nonzero()
        if miniIdx:nDimension() ~= 0 then
            miniIdx = miniIdx:select(2,1)
            local miniBatch = batchIn:index(1,miniIdx)[{{},{1,2}}]:cuda()
            local miniGrad = grad:index(1,miniIdx):cuda()
            net[i]:backward(miniBatch, miniGrad)
        end
    end
end

function netUpdateParameters(net, lr)
    net[1]:updateParameters(lr)
    net[2]:updateParameters(lr)
    net[3]:updateParameters(lr)
    net[4]:updateParameters(lr)
end


-- Define Criterion
local crit = nn.MSECriterion():cuda()

-- Define Dynamics Functions
local env = Dynamics(150)
env.width = pdata.voxel_grid:size(1)
env.length = pdata.voxel_grid:size(3)
env.height = pdata.voxel_grid:size(2)
env.epsilon = 0.3
local wshidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Do_WashCup']))
local mkidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Do_MakeHC']))
local endidx = torch.nonzero(torch.eq(pdata.SARSA_list[{{},{#rlids+1}}],actsrl['Finish']))
wshidx = wshidx[1][1]; mkidx = mkidx[1][1]; endidx = endidx[1][1]
env.wshpos = pdata.SARSA_list[{{wshidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.hcpos = pdata.SARSA_list[{{mkidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.endpos = pdata.SARSA_list[{{endidx},{idsrl['Pos_X'],idsrl['Pos_Y']}}]
env.inSize = input_size
env.outSize = output_size
print(env.wshpos, env.hcpos, env.endpos)

function env:query(state)
    local den = torch.ones(1,self.inSize)
    den[1][idsrl['Pos_X']] = self.width
    den[1][idsrl['Pos_Y']] = self.height
    qstate = torch.cdiv(state[{{},{1,2}}], den)*2-1
    local phs = state[1][idsrl['Phase']]
    return allNets[phs]:forward(qstate:cuda())
end

function env:new_state()
    local x = torch.random(1,self.width)
    local y = torch.random(1,self.length)
    local st = torch.zeros(1,self.inSize+1)
    st[1][idsrl['Pos_X']] = x
    st[1][idsrl['Pos_Y']] = y
    st[1][idsrl['Phase']] = 1
    return st
end

function env:reward(state, action, next_state)
    local x1 = state[1][idsrl['Pos_X']]
    local y1 = state[1][idsrl['Pos_Y']]
    local phs = state[1][idsrl['Phase']]
    local pos = torch.Tensor({{x1,y1}})

    local rad = 4
    local reward = 0
    if action == actsrl['Finish'] then
        if phs == 3 then
            local dist = torch.dist(pos,self.endpos)
            reward = reward + 50 * util.bool2int(dist<rad)
        else
            reward = reward - 20
        end
    elseif action == actsrl['Do_MakeHC'] then
        if phs == 2 then
            local dist = torch.dist(pos,self.hcpos)
            local isin = util.bool2int(dist<rad)
            reward = reward - 10 * math.max(dist-rad/2,0) * (1-isin) + 20 * isin
        else
            reward = reward - 20
        end
    elseif action == actsrl['Do_WashCup'] then
        if phs == 1 then
            local dist = torch.dist(pos,self.wshpos)
            local isin = util.bool2int(dist<rad)
            reward = reward - 10 * math.max(dist-rad/2,0) * (1-isin) + 20 * isin
        else
            reward = reward - 20
        end
    else
        reward = reward - 0.25*pdata.voxel_grid[{{x1},{5,10},{y1}}]:sum()
    end
    return reward
end

function env:transition(instate, action)
    local state = instate:clone()
    local a = rlacts[action]
    local x = state[1][idsrl['Pos_X']]
    local y = state[1][idsrl['Pos_Y']]
    local phs = state[1][idsrl['Phase']]
    local isFinished = false

    if a == 'Move_North' then
        state[1][idsrl['Pos_Y']] = math.max(y-1, 1)
    elseif a == 'Move_South' then
        state[1][idsrl['Pos_Y']] = math.min(y+1, self.length)
    elseif a == 'Move_East' then
        state[1][idsrl['Pos_X']] = math.min(x+1, self.width)
    elseif a == 'Move_West' then
        state[1][idsrl['Pos_X']] = math.max(x-1, 1)
    elseif a == 'Do_WashCup' then
        if state[1][idsrl['Phase']] == 1 then
            state[1][idsrl['Phase']] = 2
        end
    elseif a == 'Do_MakeHC' then
        if state[1][idsrl['Phase']] == 2 then
            state[1][idsrl['Phase']] = 3
        end
    elseif a == 'Finish' then
        if state[1][idsrl['Phase']] == 3 then
            isFinished = true
            state[1][idsrl['Phase']] = 4
        end
    end
    return state, isFinished
end

function env:explore_action(state)
    local is_greed = torch.uniform() < self.epsilon
    local phs = state[1][idsrl['Phase']]
    local vec = self:query(state)
    local act = -1

    if is_greed then
        local n = actsrl['Move_North']
        local s = actsrl['Move_South']
        local e = actsrl['Move_East']
        local w = actsrl['Move_West']
        local wc = actsrl['Do_WashCup']
        local hc = actsrl['Do_MakeHC']
        local f = actsrl['Finish']
        local v = vec[1]
        local lu = {n,s,e,w,wc,hc,f}

        local opts = torch.Tensor({v[n],v[s],v[e],v[w],v[wc],v[hc],v[f]})
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
            if phs == 1 then
                act = actsrl['Do_WashCup']
            elseif phs == 2 then
                act = actsrl['Do_MakeHC']
            elseif phs == 3 then
                act = actsrl['Finish']
            end
        end
    end
    return act
end

function env:gen_grid(s)
    local w = self.width
    local h = self.length
    inn = torch.rand(w*h,self.inSize):cuda()
    for x=0,(w-1) do
        for y=0,(h-1) do
            inn[1+x*h+y][idsrl['Pos_X']] = x
            inn[1+x*h+y][idsrl['Pos_Y']] = y
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
config.net = allNets
config.crit = crit
config.env = env

return config
