require 'pl'
require 'nn'
require 'cunn'
require 'image'
require 'explore'

local util = require 'util'

local cli_args = lapp[[
Load config file and learn qnetwork
    -i, --iterations        (default 10000)
    -m, --memory_size       (default 1000)
    -b, --batch_size        (default 100)
    -q, --net_reset         (default 30)
    -l, --learning_rate     (default 0.001)
    -p, --print_freq        (default 50)
    -r, --state_reset       (default 150)
    -g, --gamma             (default 0.95)
    -c, --comments          (default '')
    <def_file>              (string) eg. hconly.lua
    <savedir>               (string)
]]

print(tostring(cli_args))
local model = '../models/pdata3'
local fname_saveresult1 = paths.concat(cli_args.savedir, 'res_%s_%s.png')
local fname_saveinit1 = paths.concat(cli_args.savedir, 'init_%s_%s.png')
local fname_comments = paths.concat(cli_args.savedir, 'summary.txt')
local fname_network = paths.concat(cli_args.savedir, 'Qnet.th')
os.execute('mkdir -p ' .. cli_args.savedir)
file = io.open(fname_comments, 'w')
for k,v in pairs(cli_args) do
    file:write(k .. ': ' .. v .. '\n')
end
file:write(cli_args.comments .. '\n')
file:close()

rl = dofile(cli_args.def_file)
print(rl)

qwidth = rl.env.width
qheight = rl.env.length
w = pdata.voxel_grid:size(1)
h = pdata.voxel_grid:size(3)
map = torch.Tensor(w,h)
for x=1,w do
    for y=1,h do
        map[x][y] = pdata.voxel_grid[{{x},{6,13},{y}}]:max()
    end
end
small = map:min()
range = map:max() - small
util.save_vz(string.format(fname_saveinit1, 'true', 'map'), map, small, range)

for i=0,0 do
    st = torch.zeros(3)
    st[i+1] = 1
    input = rl.env:gen_grid(st)
    outt = rl.net:forward(input)
    small = outt:min()
    range = outt:max() - small
    for j=1,#rlacts do
        out = outt[{{},{j}}]:clone()
        util.save_vz(string.format(fname_saveinit1, i,j), out:view(qwidth,qheight):clone(), small, range)
    end
end

util.train_qnetwork(rl.net, rl.crit, cli_args, rl.env, fname_saveresult1, fname_network)
--util.train_basicnetwork(rl.net, rl.crit, cli_args, rl.env)
for i=0,0 do
    st = torch.zeros(3)
    st[i+1] = 1
    input = rl.env:gen_grid(st)
    outt = rl.net:forward(input)
    min = outt:min()
    range = outt:max() - min
    for j=1,#rlacts do
        out = outt[{{},{j}}]:clone()
        util.save_vz(string.format(fname_saveresult1, i,j), out:view(qwidth,qheight):clone(), min, range)
    end
end
