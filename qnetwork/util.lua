require 'cunn'

local util = {}

function util.gen_grid(w,h,xrange,yrange)
    inn = torch.rand(w*h,2):cuda()
    for x=0,(w-1) do
        for y=0,(h-1) do
            inn[1+x*w+y][1] = (x*2/w-1)*xrange/2
            inn[1+x*h+y][2] = (y*2/h-1)*yrange/2
        end
    end
    return inn
end

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
    local memory = nil
    local batch = torch.Tensor(params.batch_size, 3):cuda()
    for i=1,params.iterations do
        local input = torch.rand(3):view(1,3):cuda()*2-1


        input[1][3] = 10*util.bool2int(torch.sqrt((input[1][1]+1.0)^2+(input[1][2]+0.5)^2)<0.5)

        if not memory then
            memory = input:clone()
        end
        if i < params.memory_size then
            memory = torch.cat(memory, input,1)
        else
            idx = torch.random(1,params.memory_size)
            memory[idx] = input
        end

        ms = memory:size(1)
        bs = math.min(params.batch_size, ms)
        ridx = torch.randperm(ms)
        for s=1,bs do
            batch[s] = memory[ridx[s]]:clone()
        end

        local f = crit:forward(net:forward(batch[{{1,bs},{1,2}}]), batch[{{1,bs},{3}}])

        net:zeroGradParameters()
        local grad = crit:backward(net.output, batch[{{1,bs},{3}}])
        net:backward(batch[{{1,bs},{1,2}}],grad )
        net:updateParameters(params.learning_rate)

        if i % params.print_freq == 0 then
            print('Iter: ' .. i .. '\t Loss=' .. f)
        end
    end
end

return util
