require 'cunn'

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

function util.train_qnetwork(net, crit, params, dynamics)
    local memoryState = nil
    local memoryReward = nil
    local memoryNState = nil
    local memoryTerm = nil
    local memoryMask = nil
    local hat = net:clone()
    local batchIn = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchNIn = torch.Tensor(params.batch_size, dynamics.inSize)
    local batchReward = torch.Tensor(params.batch_size, dynamics.outSize)
    local batchTerm = torch.Tensor(params.batch_size, 1)
    local batchMask = torch.Tensor(params.batch_size, dynamics.outSize)
    for i=1,params.iterations do
        local input, outIdx, reward, new_state, terminal = dynamics:step(4)
        local output = torch.zeros(1,dynamics.outSize)
        output[1][outIdx] = reward

        local mask = torch.zeros(1,dynamics.outSize):scatter(2,torch.LongTensor{outIdx}:view(1,1),1)

        input = dynamics:normalize_points(input)
        new_state = dynamics:normalize_points(input)

        if not memoryState then
            memoryState = input:clone()
            memoryNState = new_state:clone()
            memoryReward = output:clone()
            memoryTerm = torch.Tensor(1,1):fill(util.bool2int(terminal))
            memoryMask = mask:clone()
        end
        if i < params.memory_size then
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

        ms = memoryState:size(1)
        bs = math.min(params.batch_size, ms)
        ridx = torch.randperm(ms):type('torch.LongTensor')
        batchIn = memoryState:index(1, ridx[{{1,bs}}]):cuda()
        batchNIn = memoryNState:index(1, ridx[{{1,bs}}]):cuda()
        batchReward = memoryReward:index(1, ridx[{{1,bs}}]):cuda()
        batchMask = memoryMask:index(1, ridx[{{1,bs}}]):cuda()
        batchTerm = memoryTerm:index(1, ridx[{{1,bs}}])
        batchFinal = torch.zeros(#batchReward):cuda()

        local cvals = hat:forward(batchNIn[{{1,bs},{}}])
        local nQvals, nQidx = cvals:max(2)
        nQvals = nQvals*params.gamma
        batchFinal:maskedCopy(batchMask, nQvals)
        batchFinal = batchFinal + batchReward
        local bterms = batchTerm:nonzero()
        if bterms:nDimension() ~= 0 then
            local tmp = batchTerm:repeatTensor(1,dynamics.outSize):cuda()
            local termMask = torch.cmul(tmp, batchMask)
            batchFinal:maskedCopy(termMask, batchReward:maskedSelect(termMask))
        end


        local result = net:forward(batchIn[{{1,bs},{}}])

        crit:forward(result, batchFinal[{{1,bs},{}}])

        net:zeroGradParameters()
        local grad = crit:backward(net.output, batchFinal[{{1,bs},{}}])
        grad:maskedFill(batchMask[{{1,bs},{}}]:eq(0), 0)
        net:backward(batchIn[{{1,bs},{}}], grad)
        net:updateParameters(params.learning_rate)
        local loss = grad:pow(2):sum()

        if i % params.print_freq == 0 then
            print('Iter: ' .. i .. '\t Loss=' .. loss)
        end
        if i % params.net_reset == 0 then
            hat = net:clone()
        end
    end
end

function util.train_basicnetwork(net, crit, params, dynamics)
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

function util.train_basicnetwork_backup(net, crit, params, dynamics)
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
