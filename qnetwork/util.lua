require 'cunn'

local util = {}

function util.save_vz(fname, grid)
    print('Saving ' .. fname .. ': min=' .. grid:min() .. 'max=' .. grid:max())
    im = grid
    im = im - im:min()
    im = im/im:max()
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
    local memoryIn = nil
    local memoryOut = nil
    local memoryTerm = nil
    local outMask = nil
    local hat = net:clone()
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
            outMask = mask:clone()
        end
        if i < params.memory_size then
            memoryIn = torch.cat(memoryIn, input,1)
            memoryOut = torch.cat(memoryOut, output,1)
            memoryTerm = torch.cat(memoryTerm, torch.Tensor(1,1):fill(util.bool2int(terminal)),1)
            outMask = torch.cat(outMask, mask,1)
        else
            idx = torch.random(1,params.memory_size)
            memoryIn[idx] = input
            memoryOut[idx] = output
            memoryTerm[idx] = util.bool2int(terminal)
            outMask[idx] = mask
        end

        ms = memoryIn:size(1)
        bs = math.min(params.batch_size, ms)
        ridx = torch.randperm(ms):type('torch.LongTensor')
        batchIn = memoryIn:index(1, ridx[{{1,bs}}]):cuda()
        batchOut = memoryOut:index(1, ridx[{{1,bs}}]):cuda()
        batchMask = outMask:index(1, ridx[{{1,bs}}]):cuda()

        local cvals = hat:forward(batchIn[{{1,bs},{}}])

        local result = net:forward(batchIn[{{1,bs},{}}])

        crit:forward(result, batchOut[{{1,bs},{}}])

        net:zeroGradParameters()
        local grad = crit:backward(net.output, batchOut[{{1,bs},{}}])
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

return util
