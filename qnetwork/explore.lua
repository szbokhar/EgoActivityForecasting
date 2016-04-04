local explore = {}
local Dynamics = torch.class('Dynamics')

function Dynamics:__init(reset_count)
    self.reset_count = reset_count
    self.counter = 0
    self.cstate = nil
    self.cact = nil
    self.creward = nil
    self.cnext_state = nil
end

function Dynamics:new_state()
    error('new_state() function must be implemented')
    return nil
end

function Dynamics:reward(state, action, next_state)
    error('reward() function must be implemented')
    return nil
end

function Dynamics:transition(state, action)
    error('transition() function must be implemented')
    return nil, 0
end

function Dynamics:explore_action(state)
    error('explore_action() function must be implemented')
    return nil, 0
end

function Dynamics:get_exp()
    return self.cstate:clone(), self.cact, self.creward, self.cnext_state:clone()
end

function Dynamics:step_one()
    if not self.cstate then
        self.cstate = self:new_state()
        self.counter = 0
    else
        self.cstate = self.cnext_state
    end

    self.cact = self:explore_action(self.cstate)
    self.cnext_state, reset = self:transition(self.cstate, self.cact)
    self.creward = self:reward(self.cstate, self.cact, self.cnext_state)
    self.counter = self.counter + 1

    return self.counter > self.reset_count or reset

end

function Dynamics:step(n)
    for i=0,n do
        reset = self:step_one()
        print(i)
        if reset then
            print('outt')
            break
        end
    end

    s, a, r, ns = self:get_exp()
    if reset then
        self.cstate = nil
    end

    return s, a, r, ns
end



return explore
