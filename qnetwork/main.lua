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
    -l, --learning_rate     (default 0.01)
    -p, --print_freq        (default 50)
    -r, --state_reset       (default 150)
    <def_file>              (string)
]]

print(cli_args)

rl = dofile(cli_args.def_file)


--[[
qwidth = 64
qheight = 64

inn = util.gen_grid(qwidth, qheight, 8, 8)

outt = rl.net:forward(inn)
util.save_vz(fname_saveinit, outt:view(qwidth,qheight):clone())

util.train_qnetwork(rl.net, rl.crit, cli_args, rl.env)

outt = rl.net:forward(inn)
image.save(fname_saveresult, im)
util.save_vz(fname_saveresult, outt:view(qwidth,qheight):clone())
--]]
