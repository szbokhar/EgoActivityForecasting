local io = require 'io'

local load_data = {}

function load_data.read_txt_table(fname)
    local tab = {}
    local file = io.open(fname)
    for line in file:lines() do
        local k,v = unpack(line:split(" "))
        tab[tonumber(k)+1] = v
    end
    return tab
end

return load_data
