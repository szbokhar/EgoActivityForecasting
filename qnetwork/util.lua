require 'cunn'
local matio = require 'matio'

local util = {}

function util.save_vz(fname, grid, min, range)
    print('Saving ' .. fname .. ': min=' .. grid:min() .. 'max=' .. grid:max())
    im = grid
    im = im - min
    im = im/range
    image.save(fname, im)
end

function util.bool2int(b) return b and 1 or 0 end

function util.rev_table(tbl)
    rev = {}
    for i,v in ipairs(tbl) do
        rev[v] = i
    end
    return rev
end

function util.train_qnetwork_3state(net, crit, params, dynamics, fnamesave, fname_network, fname_mat)
    local memoryState = {}
    local memoryNState = {}
    local memoryReward = nil
    local memoryTerm = nil
    local memoryMask = nil
    local termIn = {}
    local termOut = nil
    local termTerm = nil
    local termMask = nil
    local hat = netClone(net)
    local batchState = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchNState = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchTarget = torch.Tensor(params.batch_size, dynamics.outSize)
    local batchTerm = torch.Tensor(params.batch_size, 1)
    local batchMask = torch.Tensor(params.batch_size, dynamics.outSize)
    local lossvals = torch.Tensor(10)
    for i=0,params.iterations do
        local input, outIdx, reward, new_state, terminal = dynamics:step(1)
        local output = torch.zeros(1,dynamics.outSize)
        output[1][outIdx] = reward

        local mask = torch.zeros(1,dynamics.outSize):scatter(2,torch.LongTensor{outIdx}:view(1,1),1)

        input, phase = dynamics:normalize_points(input)
        new_state, new_phase = dynamics:normalize_points(new_state)

        if not terminal then
            if not memoryState[1] then
                memoryState[1] = input:clone()
                memoryState[2] = phase:clone()
                memoryNState[1] = new_state:clone()
                memoryNState[2] = new_phase:clone()
                memoryReward = output:clone()
                memoryTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
                memoryMask = mask:clone()
            end
            if memoryState[1]:size(1) < params.memory_size then
                memoryState[1] = torch.cat(memoryState[1], input,1)
                memoryState[2] = torch.cat(memoryState[2], phase,1)
                memoryNState[1] = torch.cat(memoryNState[1], new_state,1)
                memoryNState[2] = torch.cat(memoryNState[2], new_phase,1)
                memoryReward = torch.cat(memoryReward, output,1)
                memoryTerm = torch.cat(memoryTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
                memoryMask = torch.cat(memoryMask, mask,1)
            else
                idx = torch.random(1,params.memory_size)
                memoryState[1][idx] = input
                memoryState[2][idx] = phase
                memoryNState[1][idx] = new_state
                memoryNState[2][idx] = new_phase
                memoryReward[idx] = output
                memoryTerm[idx] = util.bool2int(terminal)
                memoryMask[idx] = mask
            end

        else
            if not termIn[1] then
                termIn[1] = input:clone()
                termIn[2] = phase:clone()
                termOut = output:clone()
                termTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
                termMask = mask:clone()
            end
            if termIn[1]:size(1) < params.memory_size then
                termIn[1] = torch.cat(termIn[1], input,1)
                termIn[2] = torch.cat(termIn[2], phase,1)
                termOut = torch.cat(termOut, output,1)
                termTerm = torch.cat(termTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
                termMask = torch.cat(termMask, mask,1)
            else
                idx = torch.random(1,params.memory_size)
                termIn[1][idx] = input
                termIn[2][idx] = phase
                termOut[idx] = output
                termTerm[idx] = util.bool2int(terminal)
                termMask[idx] = mask
            end
        end

        if memoryState[1] ~= nil then
            ms = memoryState[1]:size(1)
            bs = math.min(params.batch_size, ms)
            ridx = torch.randperm(ms):type('torch.LongTensor')
            batchState = {}
            batchState[1] = memoryState[1]:index(1, ridx[{{1,bs}}])
            batchState[2] = memoryState[2]:index(1, ridx[{{1,bs}}])
            batchNState = {}
            batchNState[1] = memoryNState[1]:index(1, ridx[{{1,bs}}])
            batchNState[2] = memoryNState[2]:index(1, ridx[{{1,bs}}])
            batchTarget = memoryReward:index(1, ridx[{{1,bs}}])
            batchMask = memoryMask:index(1, ridx[{{1,bs}}])
            batchFinal = torch.zeros(#batchTarget)

            local cvals = netForward(hat, batchNState[1][{{1,bs},{}}], batchNState[2][{{1,bs},{}}])
            local nQvals, nQidx = cvals:max(2)
            nQvals = nQvals*params.gamma
            batchFinal:maskedCopy(batchMask:eq(1), nQvals)
            batchFinal = batchFinal + batchTarget

            local result = netForward(net, batchState[1][{{1,bs},{}}], batchState[2][{{1,bs},{}}])

            crit:forward(result, batchFinal[{{1,bs},{}}])

            netZeroGradParameters(net)
            local grad = crit:backward(result:cuda(), batchFinal[{{1,bs},{}}]:cuda())
            grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0):cuda(), 0)
            netBackward(net, batchState[1][{{1,bs},{}}], batchState[2][{{1,bs},{}}], grad)
            netUpdateParameters(net, params.learning_rate)
            local loss = grad:pow(2):sum()

            if i % params.print_freq == 0 then
                print('Iter: ' .. i .. '\t Loss=' .. loss)
                lossvals = torch.cat(lossvals, torch.Tensor{loss},1)
            end
            outt = {}
            tomat = torch.Tensor(qwidth, qheight, #net, #rlacts)
            if i % (400*params.print_freq) == 0 or i == params.iterations then
                for phs=1,#net do
                    st = torch.zeros(3)
                    input = rl.env:gen_grid(st)
                    outt['act'..phs] = rl.net[phs]:forward(input)
                    outt['act'..phs] = outt['act'..phs]:type('torch.DoubleTensor')
                    min = outt['act'..phs]:min()
                    range = outt['act'..phs]:max() - min
                    for j=1,#rlacts do
                        out = outt['act'..phs][{{},{j}}]:clone()
                        tomat[{{},{},phs,j}] = out:view(qwidth,qheight)
                        util.save_vz(string.format(fnamesave,i..'-'..phs,j), out:view(qwidth,qheight):clone(), min, range)
                    end
                    torch.save(string.format(fname_network, phs), hat)

                end
                matio.save(fname_mat, {Q=tomat, vals=lossvals, umap=torch.Tensor{10}, voxel_grid=pdata.voxel_grid:type('torch.DoubleTensor')})
                print('---> ' .. os.date())
            end
        end

        if termIn[1] ~= nil then
            ms = termIn[1]:size(1)
            bs = math.min(params.batch_size, ms)
            ridx = torch.randperm(ms):type('torch.LongTensor')
            batchState = {}
            batchState[1] = termIn[1]:index(1, ridx[{{1,bs}}])
            batchState[2] = termIn[2]:index(1, ridx[{{1,bs}}])
            batchTarget = termOut:index(1, ridx[{{1,bs}}])
            batchMask = termMask:index(1, ridx[{{1,bs}}])

            local result = netForward(net, batchState[1][{{1,bs},{}}], batchState[2][{{1,bs},{}}])

            crit:forward(result, batchTarget[{{1,bs},{}}])

            netZeroGradParameters(net)
            local grad = crit:backward(result:cuda(), batchTarget[{{1,bs},{}}]:cuda())
            grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0):cuda(), 0)
            netBackward(net, batchState[1][{{1,bs},{}}], batchState[2][{{1,bs},{}}], grad)
            netUpdateParameters(net, params.learning_rate)
        end

        if i % params.net_reset == 0 then
            hat = netClone(net)
        end
    end
end

function util.train_qnetwork(net, crit, params, dynamics, fnamesave, fname_network)
    local memoryState = nil
    local memoryNState = nil
    local memoryReward = nil
    local memoryTerm = nil
    local memoryMask = nil
    local termIn = nil
    local termOut = nil
    local termTerm = nil
    local termMask = nil
    local hat = net:clone()
    local batchState = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchNState = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchTarget = torch.Tensor(params.batch_size, dynamics.outSize)
    local batchTerm = torch.Tensor(params.batch_size, 1)
    local batchMask = torch.Tensor(params.batch_size, dynamics.outSize)
    for i=1,params.iterations do
        local input, outIdx, reward, new_state, terminal = dynamics:step(1)
        local output = torch.zeros(1,dynamics.outSize)
        output[1][outIdx] = reward

        local mask = torch.zeros(1,dynamics.outSize):scatter(2,torch.LongTensor{outIdx}:view(1,1),1)

        input = dynamics:normalize_points(input)
        new_state = dynamics:normalize_points(new_state)

        if not terminal then
            if not memoryState then
                memoryState = input:clone()
                memoryNState = new_state:clone()
                memoryReward = output:clone()
                memoryTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
                memoryMask = mask:clone()
            end
            if memoryState:size(1) < params.memory_size then
                memoryState = torch.cat(memoryState, input,1)
                memoryNState = torch.cat(memoryNState, new_state,1)
                memoryReward = torch.cat(memoryReward, output,1)
                memoryTerm = torch.cat(memoryTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
                memoryMask = torch.cat(memoryMask, mask,1)
            else
                idx = torch.random(1,params.memory_size)
                memoryState[idx] = input
                memoryNState[idx] = new_state
                memoryReward[idx] = output
                memoryTerm[idx] = util.bool2int(terminal)
                memoryMask[idx] = mask
            end

        else
            if not termIn then
                termIn = input:clone()
                termOut = output:clone()
                termTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
                termMask = mask:clone()
            end
            if termIn:size(1) < params.memory_size then
                termIn = torch.cat(termIn, input,1)
                termOut = torch.cat(termOut, output,1)
                termTerm = torch.cat(termTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
                termMask = torch.cat(termMask, mask,1)
            else
                idx = torch.random(1,params.memory_size)
                termIn[idx] = input
                termOut[idx] = output
                termTerm[idx] = util.bool2int(terminal)
                termMask[idx] = mask
            end
        end

        if memoryState ~= nil then
            ms = memoryState:size(1)
            bs = math.min(params.batch_size, ms)
            ridx = torch.randperm(ms):type('torch.LongTensor')
            batchState = memoryState:index(1, ridx[{{1,bs}}]):cuda()
            batchNState = memoryNState:index(1, ridx[{{1,bs}}]):cuda()
            batchTarget = memoryReward:index(1, ridx[{{1,bs}}]):cuda()
            batchMask = memoryMask:index(1, ridx[{{1,bs}}]):cuda()
            batchFinal = torch.zeros(#batchTarget):cuda()

            local cvals = hat:forward(batchNState[{{1,bs},{}}])
            local nQvals, nQidx = cvals:max(2)
            nQvals = nQvals*params.gamma
            batchFinal:maskedCopy(batchMask:eq(1), nQvals)
            batchFinal = batchFinal + batchTarget

            local result = net:forward(batchState[{{1,bs},{}}])

            crit:forward(result, batchFinal[{{1,bs},{}}])

            net:zeroGradParameters()
            local grad = crit:backward(net.output, batchFinal[{{1,bs},{}}])
            grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0), 0)
            net:backward(batchState[{{1,bs},{}}], grad)
            net:updateParameters(params.learning_rate)
            local loss = grad:pow(2):sum()

            if i % params.print_freq == 0 then
                print('Iter: ' .. i .. '\t Loss=' .. loss)
            end
            if i % (100*params.print_freq) == 0 then
                st = torch.zeros(3)
                input = rl.env:gen_grid(st)
                outt = rl.net:forward(input)
                min = outt:min()
                range = outt:max() - min
                for j=1,#rlacts do
                    out = outt[{{},{j}}]:clone()
                    util.save_vz(string.format(fnamesave, i,j), out:view(qwidth,qheight):clone(), min, range)
                end
                torch.save(fname_network, hat)
            end
        end

        if termIn ~= nil then
            ms = termIn:size(1)
            bs = math.min(params.batch_size, ms)
            ridx = torch.randperm(ms):type('torch.LongTensor')
            batchState = termIn:index(1, ridx[{{1,bs}}]):cuda()
            batchTarget = termOut:index(1, ridx[{{1,bs}}]):cuda()
            batchMask = termMask:index(1, ridx[{{1,bs}}]):cuda()

            local result = net:forward(batchState[{{1,bs},{}}])

            crit:forward(result, batchTarget[{{1,bs},{}}])

            net:zeroGradParameters()
            local grad = crit:backward(net.output, batchTarget[{{1,bs},{}}])
            grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0), 0)
            net:backward(batchState[{{1,bs},{}}], grad)
            net:updateParameters(params.learning_rate)
        end

        if i % params.net_reset == 0 then
            hat = net:clone()
        end
    end
end

function util.train_basicnetwork(net, crit, params, dynamics)
    local memoryIn = nil
    local memoryOut = nil
    local memoryTerm = nil
    local memoryMask = nil
    local batchIn = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchOut = torch.Tensor(params.batch_size, dynamics.outSize)
    local batchTerm = torch.Tensor(params.batch_size, 1)
    local batchMask = torch.Tensor(params.batch_size, dynamics.outSize)
    for i=1,params.iterations do
        local input, outIdx, reward, new_state, terminal = dynamics:step(1)
        local output = torch.zeros(1,dynamics.outSize)
        output[1][outIdx] = reward

        local mask = torch.zeros(1,dynamics.outSize):scatter(2,torch.LongTensor{outIdx}:view(1,1),1)

        input = dynamics:normalize_points(input)

        if not memoryIn then
            memoryIn = input:clone()
            memoryOut = output:clone()
            memoryTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
            memoryMask = mask:clone()
        end
        if i < params.memory_size then
            memoryIn = torch.cat(memoryIn, input,1)
            memoryOut = torch.cat(memoryOut, output,1)
            memoryTerm = torch.cat(memoryTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
            memoryMask = torch.cat(memoryMask, mask,1)
        else
            idx = torch.random(1,params.memory_size)
            memoryIn[idx] = input
            memoryOut[idx] = output
            memoryTerm[idx] = util.bool2int(terminal)
            memoryMask[idx] = mask
        end

        ms = memoryIn:size(1)
        bs = math.min(params.batch_size, ms)
        ridx = torch.randperm(ms):type('torch.LongTensor')
        batchIn = memoryIn:index(1, ridx[{{1,bs}}]):cuda()
        batchOut = memoryOut:index(1, ridx[{{1,bs}}]):cuda()
        batchMask = memoryMask:index(1, ridx[{{1,bs}}]):cuda()

        local result = net:forward(batchIn[{{1,bs},{}}])

        crit:forward(result, batchOut[{{1,bs},{}}])

        net:zeroGradParameters()
        local grad = crit:backward(net.output, batchOut[{{1,bs},{}}])
        grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0), 0)
        net:backward(batchIn[{{1,bs},{}}], grad)
        net:updateParameters(params.learning_rate)
        local loss = grad:pow(2):sum()


        --[[
        termSize = memoryTerm:sum()
        if termSize > 0 then
            ms = memoryIn:size(1)
            bs = math.min(params.batch_size, termSize)
            c = torch.randperm(termSize):type('torch.LongTensor')
            ridx = memoryTerm:nonzero():select(2,1):index(1, c[{{1,bs}}])
            batchIn = memoryIn:index(1, ridx[{{1,bs}}]):cuda()
            batchOut = memoryOut:index(1, ridx[{{1,bs}}]):cuda()
            batchMask = memoryMask:index(1, ridx[{{1,bs}}]):cuda()

            local result = net:forward(batchIn[{{1,bs},{}}])

            crit:forward(result, batchOut[{{1,bs},{}}])

            net:zeroGradParameters()
            local grad = crit:backward(net.output, batchOut[{{1,bs},{}}])
            grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0), 0)
            net:backward(batchIn[{{1,bs},{}}], grad)
            net:updateParameters(params.learning_rate)
        end
        --]]




        if i % params.print_freq == 0 then
            print('Iter: ' .. i .. '\t Loss=' .. loss)
        end
    end
end
return util
